//! Platform-specific system audio capture (ScreenCaptureKit / WASAPI loopback / PipeWire).

use crossbeam_channel::Sender;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;

/// Join handle for the capture thread; dropping waits for clean shutdown.
pub struct SystemCaptureHandle {
    _thread: thread::JoinHandle<()>,
}

pub struct SystemCaptureInfo {
    pub sample_rate: u32,
    pub backend_name: &'static str,
}

/// True when a backend is compiled for this target.
pub fn system_audio_supported() -> bool {
    cfg!(target_os = "macos")
        || cfg!(target_os = "windows")
        || cfg!(all(target_os = "linux", feature = "linux-system-audio"))
}

/// Start capturing system audio after preflight; sends mono `f32` at native rate until `stop`.
pub fn start_system_capture(
    stop: Arc<AtomicBool>,
    tx: Sender<f32>,
) -> anyhow::Result<(SystemCaptureHandle, SystemCaptureInfo)> {
    #[cfg(target_os = "macos")]
    {
        start_macos(stop, tx)
    }
    #[cfg(target_os = "windows")]
    {
        start_windows(stop, tx)
    }
    #[cfg(all(target_os = "linux", feature = "linux-system-audio"))]
    {
        start_linux(stop, tx)
    }
    #[cfg(all(target_os = "linux", not(feature = "linux-system-audio")))]
    {
        let _ = (&stop, &tx);
        anyhow::bail!(
            "Linux system audio was disabled at compile time (e.g. `cross` Linux builds). \
             Build natively on Linux with default features, or pass `--features linux-system-audio` \
             when libpipewire-0.3-dev and libspa-0.2-dev are installed."
        );
    }
    #[cfg(not(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux"
    )))]
    {
        anyhow::bail!("system audio is not supported on this platform")
    }
}

// --- macOS ---

#[cfg(target_os = "macos")]
fn start_macos(
    stop: Arc<AtomicBool>,
    tx: Sender<f32>,
) -> anyhow::Result<(SystemCaptureHandle, SystemCaptureInfo)> {
    use anyhow::Context;
    use screencapturekit::cm::CMSampleBuffer;
    use screencapturekit::prelude::*;
    use screencapturekit::stream::output_trait::SCStreamOutputTrait;
    use screencapturekit::stream::output_type::SCStreamOutputType;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    const SAMPLE_RATE: i32 = 48_000;

    let content = SCShareableContent::get()
        .map_err(|e| anyhow::anyhow!(e))
        .with_context(|| {
            "ScreenCaptureKit could not list displays (grant Screen Recording in System Settings > Privacy & Security)"
        })?;
    let displays = content.displays();
    let display = displays
        .first()
        .context("no displays available for system audio capture")?;

    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();

    let config = SCStreamConfiguration::new()
        .with_width(2)
        .with_height(2)
        .with_captures_audio(true)
        .with_sample_rate(SAMPLE_RATE)
        .with_channel_count(1);

    struct AudioOut {
        tx: Sender<f32>,
        stop: Arc<AtomicBool>,
        format_checked: std::sync::atomic::AtomicBool,
    }

    impl SCStreamOutputTrait for AudioOut {
        fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
            if of_type != SCStreamOutputType::Audio {
                return;
            }
            if self.stop.load(Ordering::SeqCst) {
                return;
            }
            let Some(buf_list) = sample.audio_buffer_list() else {
                return;
            };
            for i in 0..buf_list.num_buffers() {
                let Some(buf_ref) = buf_list.buffer(i) else {
                    continue;
                };
                let bytes = buf_ref.data();
                if !bytes.is_empty()
                    && !self.format_checked.swap(true, Ordering::SeqCst)
                    && bytes.len() % 4 != 0
                {
                    eprintln!(
                        "active-listener: unexpected ScreenCaptureKit audio buffer size (not multiple of 4); check format"
                    );
                }
                for chunk in bytes.chunks_exact(4) {
                    let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if self.tx.send(s).is_err() {
                        return;
                    }
                }
            }
        }
    }

    let handler = AudioOut {
        tx: tx.clone(),
        stop: stop.clone(),
        format_checked: std::sync::atomic::AtomicBool::new(false),
    };

    let jh = thread::spawn(move || {
        let run = || -> anyhow::Result<()> {
            let mut stream = SCStream::new(&filter, &config);
            stream.add_output_handler(handler, SCStreamOutputType::Audio);
            stream.start_capture().map_err(|e| anyhow::anyhow!(e))?;
            while !stop.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(50));
            }
            let _ = stream.stop_capture();
            Ok(())
        };
        if let Err(e) = run() {
            eprintln!("active-listener: system audio capture stopped: {e:#}");
        }
    });

    Ok((
        SystemCaptureHandle { _thread: jh },
        SystemCaptureInfo {
            sample_rate: SAMPLE_RATE as u32,
            backend_name: "screencapturekit",
        },
    ))
}

// --- Windows ---

#[cfg(target_os = "windows")]
fn start_windows(
    stop: Arc<AtomicBool>,
    tx: Sender<f32>,
) -> anyhow::Result<(SystemCaptureHandle, SystemCaptureInfo)> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;
    use wasapi::{initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode};

    let com_init = initialize_mta();
    if com_init.is_err() {
        anyhow::bail!("WASAPI COM (MTA) init failed: {com_init:?}");
    }

    let rate_out = Arc::new(AtomicU32::new(0));
    let rate_for_thread = rate_out.clone();

    let jh = thread::spawn(move || {
        let r = (|| -> anyhow::Result<()> {
            let enumerator = DeviceEnumerator::new().map_err(|e| anyhow::anyhow!(e))?;
            let device = enumerator
                .get_default_device(&Direction::Render)
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut audio_client = device
                .get_iaudioclient()
                .map_err(|e| anyhow::anyhow!(e))?;
            let mix_format = audio_client.get_mixformat().map_err(|e| anyhow::anyhow!(e))?;
            rate_for_thread.store(mix_format.get_samplespersec(), Ordering::SeqCst);
            let channels = mix_format.get_nchannels() as usize;
            let subformat = mix_format.get_subformat().map_err(|e| anyhow::anyhow!(e))?;
            let blockalign = mix_format.get_blockalign() as usize;

            let mode = StreamMode::PollingShared {
                autoconvert: true,
                buffer_duration_hns: 200_000,
            };
            audio_client
                .initialize_client(&mix_format, &Direction::Capture, &mode)
                .map_err(|e| anyhow::anyhow!(e))?;
            let capture = audio_client
                .get_audiocaptureclient()
                .map_err(|e| anyhow::anyhow!(e))?;
            audio_client.start_stream().map_err(|e| anyhow::anyhow!(e))?;

            let buffer_frames = audio_client.get_buffer_size().map_err(|e| anyhow::anyhow!(e))? as usize;
            let mut buf = vec![0u8; buffer_frames * blockalign.max(1)];

            while !stop.load(Ordering::SeqCst) {
                match capture.read_from_device(&mut buf) {
                    Ok((frames, _)) => {
                        if frames == 0 {
                            thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        let n_bytes = frames as usize * blockalign;
                        let data = &buf[..n_bytes];
                        match subformat {
                            SampleType::Float => {
                                let frame_bytes = channels * 4;
                                for frame in data.chunks_exact(frame_bytes) {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        let off = ch * 4;
                                        let s = f32::from_le_bytes([
                                            frame[off],
                                            frame[off + 1],
                                            frame[off + 2],
                                            frame[off + 3],
                                        ]);
                                        sum += s;
                                    }
                                    let _ = tx.send(sum / channels as f32);
                                }
                            }
                            SampleType::Int => {
                                let frame_bytes = channels * 2;
                                for frame in data.chunks_exact(frame_bytes) {
                                    let mut sum = 0.0f32;
                                    for ch in 0..channels {
                                        let off = ch * 2;
                                        let v = i16::from_le_bytes([frame[off], frame[off + 1]]);
                                        sum += v as f32 / 32768.0;
                                    }
                                    let _ = tx.send(sum / channels as f32);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("active-listener: WASAPI loopback read error: {e}");
                        break;
                    }
                }
            }
            let _ = audio_client.stop_stream();
            Ok(())
        })();
        if let Err(e) = r {
            eprintln!("active-listener: system audio capture stopped: {e:#}");
        }
    });

    let mut sample_rate = 48_000u32;
    for _ in 0..80 {
        let r = rate_out.load(Ordering::SeqCst);
        if r != 0 {
            sample_rate = r;
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }

    Ok((
        SystemCaptureHandle { _thread: jh },
        SystemCaptureInfo {
            sample_rate,
            backend_name: "wasapi-loopback",
        },
    ))
}

// --- Linux (PipeWire) ---

#[cfg(all(target_os = "linux", feature = "linux-system-audio"))]
fn start_linux(
    stop: Arc<AtomicBool>,
    tx: Sender<f32>,
) -> anyhow::Result<(SystemCaptureHandle, SystemCaptureInfo)> {
    use anyhow::Context;
    use pipewire as pw;
    use pw::properties::properties;
    use pw::spa;
    use pw::spa::param::format::{MediaSubtype, MediaType};
    use pw::spa::param::format_utils;
    use pw::spa::pod::Pod;
    use std::mem;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    let rate_out = Arc::new(AtomicU32::new(0));
    let rate_wait = rate_out.clone();

    let jh = thread::spawn(move || {
        let run = || -> anyhow::Result<()> {
            pw::init();

            let mainloop = pw::main_loop::MainLoopRc::new(None)?;
            let ml_quit = mainloop.clone();
            let stop_watcher = stop.clone();
            thread::spawn(move || {
                while !stop_watcher.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(100));
                }
                ml_quit.quit();
            });

            let context = pw::context::ContextRc::new(&mainloop, None)?;
            let core = context
                .connect_rc(None)
                .context("connect to PipeWire (is the daemon running?)")?;

            let props = properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Capture",
                *pw::keys::MEDIA_ROLE => "Music",
                *pw::keys::STREAM_CAPTURE_SINK => "true",
            };

            let stream = pw::stream::StreamBox::new(&core, "active-listener-system", props)?;

            struct UserData {
                tx: Sender<f32>,
                format: spa::param::audio::AudioInfoRaw,
            }

            let rate_for_cb = rate_out.clone();
            let user_data = UserData {
                tx,
                format: Default::default(),
            };

            let _listener = stream
                .add_local_listener_with_user_data(user_data)
                .param_changed(move |_, user_data, id, param| {
                    let Some(param) = param else {
                        return;
                    };
                    if id != pw::spa::param::ParamType::Format.as_raw() {
                        return;
                    }
                    let Ok((media_type, media_subtype)) = format_utils::parse_format(param) else {
                        return;
                    };
                    if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                        return;
                    }
                    if user_data.format.parse(param).is_ok() {
                        let r = user_data.format.rate();
                        if r > 0 {
                            rate_for_cb.store(r, Ordering::SeqCst);
                        }
                    }
                })
                .process(|stream, user_data| {
                    let Some(mut buffer) = stream.dequeue_buffer() else {
                        return;
                    };
                    let datas = buffer.datas_mut();
                    if datas.is_empty() {
                        return;
                    }
                    let data = &mut datas[0];
                    let n_channels = user_data.format.channels().max(1) as usize;
                    let n_floats = data.chunk().size() as usize / mem::size_of::<f32>();
                    let Some(samples) = data.data() else {
                        return;
                    };
                    for frame_start in (0..n_floats).step_by(n_channels) {
                        if frame_start + n_channels > n_floats {
                            break;
                        }
                        let mut sum = 0.0f32;
                        for c in 0..n_channels {
                            let idx = (frame_start + c) * mem::size_of::<f32>();
                            if idx + 4 > samples.len() {
                                return;
                            }
                            let chunk: [u8; 4] = samples[idx..idx + 4].try_into().unwrap_or([0; 4]);
                            sum += f32::from_le_bytes(chunk);
                        }
                        let _ = user_data.tx.send(sum / n_channels as f32);
                    }
                })
                .register()?;

            let mut audio_info = spa::param::audio::AudioInfoRaw::new();
            audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
            let obj = pw::spa::pod::Object {
                type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
                id: pw::spa::param::ParamType::EnumFormat.as_raw(),
                properties: audio_info.into(),
            };
            let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
                std::io::Cursor::new(Vec::new()),
                &pw::spa::pod::Value::Object(obj),
            )
            .map_err(|e| anyhow::anyhow!("serialize audio format pod: {e}"))?
            .0
            .into_inner();

            let mut params = [Pod::from_bytes(&values)?];

            stream.connect(
                spa::utils::Direction::Input,
                None,
                pw::stream::StreamFlags::AUTOCONNECT
                    | pw::stream::StreamFlags::MAP_BUFFERS
                    | pw::stream::StreamFlags::RT_PROCESS,
                &mut params,
            )?;

            mainloop.run();
            Ok(())
        };
        if let Err(e) = run() {
            eprintln!("active-listener: PipeWire system audio stopped: {e:#}");
        }
    });

    let mut sample_rate = 48_000u32;
    for _ in 0..80 {
        let r = rate_wait.load(Ordering::SeqCst);
        if r != 0 {
            sample_rate = r;
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }

    Ok((
        SystemCaptureHandle { _thread: jh },
        SystemCaptureInfo {
            sample_rate,
            backend_name: "pipewire",
        },
    ))
}
