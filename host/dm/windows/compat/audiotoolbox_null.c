// Windows port: the AudioToolbox entry points hle_audio.c uses, on WASAPI. The output unit is the default render device in shared mode, event driven: a thread of
// its own waits for the device's buffer event and pulls the render callback for the frames the device has room for, in the stream format hle_audio.c set (interleaved
// 16-bit PCM at 48 kHz, 1 or 2 channels; WASAPI converts to the device's mix format: AUTOCONVERTPCM). A device that goes away (unplugged, the default changed) is
// opened again on the next default device. Start reports the device's own failure, so hle_audio.c's silent sink stands in and retries as it does on the Mac.
// (The file keeps its stage 2 name: it held the stubs.) Real output is on with DM_AUDIO=1 (RUN_DK_WIN.cmd); off, the host's silent sink runs and nothing here is called.
// The movie library's AAC converter is still refused (kAudio_UnimplementedError): movies play without sound.
#define WIN32_LEAN_AND_MEAN
#define COBJMACROS
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <AudioToolbox/AudioToolbox.h>

#define ALOG(...) fprintf(stderr, "[dm] " __VA_ARGS__)   // host/dm.h LOG, which this file does not include
static const GUID k_clsid_enum = {0xBCDE0395, 0xE52F, 0x467C, {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};
static const GUID k_iid_enum = {0xA95664D2, 0x9614, 0x4F35, {0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}};
static const GUID k_iid_client = {0x1CB9AD4C, 0xDBFA, 0x4C32, {0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}};
static const GUID k_iid_render = {0xF294ACFC, 0x3146, 0x4483, {0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2}};

struct dm_audio_component { int unused; };
static struct dm_audio_component g_output;
struct dm_audio_unit {
    AudioStreamBasicDescription fmt; AURenderCallbackStruct cb; int has_fmt, has_cb;
    HANDLE th, stop, ready; volatile LONG run; HRESULT open_hr;
};

AudioComponent AudioComponentFindNext(AudioComponent after, const AudioComponentDescription *d) {
    return !after && d && d->componentType == kAudioUnitType_Output ? &g_output : NULL;
}
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *out) {
    if (!out) return kAudio_UnimplementedError;
    *out = c == &g_output ? calloc(1, sizeof **out) : NULL;
    return *out ? noErr : kAudio_UnimplementedError;
}
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement el, const void *data, UInt32 size) {
    (void)el;
    if (!u || !data) return kAudio_UnimplementedError;
    if (id == kAudioUnitProperty_StreamFormat && scope == kAudioUnitScope_Input && size >= sizeof u->fmt) {
        const AudioStreamBasicDescription *f = data;
        if (f->mFormatID != kAudioFormatLinearPCM || (f->mFormatFlags & (kAudioFormatFlagIsNonInterleaved | kAudioFormatFlagIsBigEndian)) ||
            !f->mChannelsPerFrame || f->mChannelsPerFrame > 8 || (f->mBitsPerChannel != 16 && f->mBitsPerChannel != 32)) return kAudioUnitErr_FormatNotSupported;
        u->fmt = *f; u->has_fmt = 1; return noErr; }
    if (id == kAudioUnitProperty_SetRenderCallback && size >= sizeof u->cb) { memcpy(&u->cb, data, sizeof u->cb); u->has_cb = 1; return noErr; }
    return kAudio_UnimplementedError;
}
OSStatus AudioUnitInitialize(AudioUnit u) { return u && u->has_fmt && u->has_cb ? noErr : kAudio_UnimplementedError; }

// The render thread: opens the default device, signals Start with the result, then fills the device buffer on each event until stopped. A device lost on the way
// (AUDCLNT_E_DEVICE_INVALIDATED) is opened again after a pause; while none can be opened the callback is not called (hle_audio.c's watchdog notices)
static DWORD WINAPI render_main(void *p) {
    AudioUnit u = p;
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    HANDLE ev = CreateEventW(NULL, FALSE, FALSE, NULL);
    const UInt32 ch = u->fmt.mChannelsPerFrame, bpf = u->fmt.mBytesPerFrame ? u->fmt.mBytesPerFrame : ch * (u->fmt.mBitsPerChannel / 8);
    WAVEFORMATEX wf = {u->fmt.mFormatFlags & kAudioFormatFlagIsFloat ? 3 /* WAVE_FORMAT_IEEE_FLOAT */ : WAVE_FORMAT_PCM, (WORD)ch, (DWORD)u->fmt.mSampleRate,
                       (DWORD)(u->fmt.mSampleRate * bpf), (WORD)bpf, (WORD)u->fmt.mBitsPerChannel, 0};
    double sample_time = 0; int first = 1, said_lost = 0;
    while (InterlockedCompareExchange(&u->run, 0, 0)) {
        IMMDeviceEnumerator *en = NULL; IMMDevice *dev = NULL; IAudioClient *cl = NULL; IAudioRenderClient *rc = NULL; UINT32 nbuf = 0;
        HRESULT hr = CoCreateInstance(&k_clsid_enum, NULL, CLSCTX_ALL, &k_iid_enum, (void **)&en);
        if (SUCCEEDED(hr)) hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(en, eRender, eConsole, &dev);
        if (SUCCEEDED(hr)) hr = IMMDevice_Activate(dev, &k_iid_client, CLSCTX_ALL, NULL, (void **)&cl);
        if (SUCCEEDED(hr)) hr = IAudioClient_Initialize(cl, AUDCLNT_SHAREMODE_SHARED,
                                   AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, 0, 0, &wf, NULL);
        if (SUCCEEDED(hr)) hr = IAudioClient_SetEventHandle(cl, ev);
        if (SUCCEEDED(hr)) hr = IAudioClient_GetBufferSize(cl, &nbuf);
        if (SUCCEEDED(hr)) hr = IAudioClient_GetService(cl, &k_iid_render, (void **)&rc);
        if (SUCCEEDED(hr)) { BYTE *b; if (SUCCEEDED(IAudioClient_GetCurrentPadding(cl, &(UINT32){0})) && SUCCEEDED(IAudioRenderClient_GetBuffer(rc, nbuf, &b)))
                                 IAudioRenderClient_ReleaseBuffer(rc, nbuf, AUDCLNT_BUFFERFLAGS_SILENT);   // start from a full buffer of silence
                             hr = IAudioClient_Start(cl); }
        if (first) { u->open_hr = hr; first = 0; SetEvent(u->ready); }
        if (SUCCEEDED(hr)) {
            REFERENCE_TIME per = 0; IAudioClient_GetDevicePeriod(cl, &per, NULL);
            ALOG("audio: WASAPI shared output, %u ch %u-bit at %.0f Hz, device buffer %u frames, period %.1f ms%s\n", (unsigned)ch, (unsigned)u->fmt.mBitsPerChannel,
                   u->fmt.mSampleRate, (unsigned)nbuf, per / 10000.0, said_lost ? " (device opened again)" : "");
            HANDLE w[2] = {ev, u->stop};
            while (InterlockedCompareExchange(&u->run, 0, 0)) {
                DWORD r = WaitForMultipleObjects(2, w, FALSE, 500);
                if (r == WAIT_OBJECT_0 + 1) break;
                UINT32 pad = 0; hr = IAudioClient_GetCurrentPadding(cl, &pad); if (FAILED(hr)) break;
                UINT32 n = nbuf - pad; if (!n) continue;
                BYTE *b = NULL; hr = IAudioRenderClient_GetBuffer(rc, n, &b); if (FAILED(hr)) break;
                AudioBufferList bl = {1, {{ch, n * bpf, b}}}; AudioUnitRenderActionFlags fl = 0;
                AudioTimeStamp ts = {sample_time, 0, 1.0, 0, 1, 0}; sample_time += n;
                u->cb.inputProc(u->cb.inputProcRefCon, &fl, &ts, 0, n, &bl);
                IAudioRenderClient_ReleaseBuffer(rc, n, 0);
            }
            IAudioClient_Stop(cl);
        }
        if (rc) IAudioRenderClient_Release(rc); if (cl) IAudioClient_Release(cl); if (dev) IMMDevice_Release(dev); if (en) IMMDeviceEnumerator_Release(en);
        if (FAILED(hr) && InterlockedCompareExchange(&u->run, 0, 0)) {
            if (!said_lost++) ALOG("[RED] audio: WASAPI output stopped or could not open (0x%08lx): opened again on the default device\n", (unsigned long)hr);
            if (WaitForSingleObject(u->stop, 1000) == WAIT_OBJECT_0) break;
        }
    }
    CloseHandle(ev); CoUninitialize();
    return 0;
}
static void unit_stop(AudioUnit u) {
    if (!u->th) return;
    InterlockedExchange(&u->run, 0); SetEvent(u->stop);
    WaitForSingleObject(u->th, INFINITE); CloseHandle(u->th); CloseHandle(u->stop); CloseHandle(u->ready); u->th = u->stop = u->ready = NULL;
}
OSStatus AudioOutputUnitStart(AudioUnit u) {
    if (!u || !u->has_fmt || !u->has_cb) return kAudio_UnimplementedError;
    if (u->th) return noErr;
    u->stop = CreateEventW(NULL, TRUE, FALSE, NULL); u->ready = CreateEventW(NULL, TRUE, FALSE, NULL); u->run = 1; u->open_hr = E_FAIL;
    u->th = CreateThread(NULL, 0, render_main, u, 0, NULL);
    if (!u->th) { CloseHandle(u->stop); CloseHandle(u->ready); u->stop = u->ready = NULL; return kAudio_UnimplementedError; }
    WaitForSingleObject(u->ready, 5000);
    if (FAILED(u->open_hr)) { ALOG("[RED] audio: WASAPI output did not open (0x%08lx)\n", (unsigned long)u->open_hr); unit_stop(u); return kAudio_UnimplementedError; }
    return noErr;
}
OSStatus AudioOutputUnitStop(AudioUnit u) { if (u) unit_stop(u); return noErr; }
OSStatus AudioUnitUninitialize(AudioUnit u) { if (u) unit_stop(u); return noErr; }
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u) { if (u) { unit_stop(u); free(u); } return noErr; }

OSStatus AudioConverterNew(const AudioStreamBasicDescription *in, const AudioStreamBasicDescription *out, AudioConverterRef *c) { (void)in; (void)out; if (c) *c = NULL; return kAudio_UnimplementedError; }
OSStatus AudioConverterSetProperty(AudioConverterRef c, AudioConverterPropertyID id, UInt32 size, const void *data) { (void)c; (void)id; (void)size; (void)data; return kAudio_UnimplementedError; }
OSStatus AudioConverterFillComplexBuffer(AudioConverterRef c, AudioConverterComplexInputDataProc proc, void *ref, UInt32 *packets, AudioBufferList *out, AudioStreamPacketDescription *desc) {
    (void)c; (void)proc; (void)ref; (void)out; (void)desc; if (packets) *packets = 0; return kAudio_UnimplementedError; }
OSStatus AudioConverterDispose(AudioConverterRef c) { (void)c; return noErr; }
