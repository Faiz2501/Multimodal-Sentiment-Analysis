import { useEffect, useRef, useState } from 'react'

type Health = { status:string; model_ready:boolean; device:string; asr_model:string }
type Analysis = { sentiment:string; sentiment_score:number; emotions:Record<string,number>; sarcasm_probability:number|null }
type HistoryItem = { id:number; text:string; sentiment:string; score:number; time:string }
const API='/api'

async function apiData<T>(response:Response):Promise<T>{
  const data=await response.json().catch(()=>({detail:'The server returned an unreadable response'}))
  if(!response.ok) throw new Error(data.detail || `Request failed (${response.status})`)
  return data as T
}

export default function App(){
  const [health,setHealth]=useState<Health|null>(null)
  const [audio,setAudio]=useState<File|null>(null)
  const [transcript,setTranscript]=useState('')
  const [analysis,setAnalysis]=useState<Analysis|null>(null)
  const [history,setHistory]=useState<HistoryItem[]>([])
  const [busy,setBusy]=useState<'transcribe'|'analyze'|null>(null)
  const [error,setError]=useState('')
  const [recording,setRecording]=useState(false)
  const [audioUrl,setAudioUrl]=useState('')
  const recorder=useRef<MediaRecorder|null>(null)
  const chunks=useRef<Blob[]>([])
  const stream=useRef<MediaStream|null>(null)
  const input=useRef<HTMLInputElement|null>(null)

  useEffect(()=>{fetch(`${API}/health`).then(apiData<Health>).then(setHealth).catch(()=>setHealth({status:'offline',model_ready:false,device:'unknown',asr_model:'openai/whisper-tiny'}))},[])
  useEffect(()=>{if(!audio){setAudioUrl('');return} const url=URL.createObjectURL(audio); setAudioUrl(url); return()=>URL.revokeObjectURL(url)},[audio])
  useEffect(()=>()=>{stream.current?.getTracks().forEach(track=>track.stop())},[])

  async function transcribe(){
    if(!audio)return
    setError(''); setBusy('transcribe')
    try{const form=new FormData();form.append('audio',audio);const result=await apiData<{text:string}>(await fetch(`${API}/transcribe`,{method:'POST',body:form}));setTranscript(result.text)}
    catch(e){setError(e instanceof Error?e.message:'Transcription failed')}finally{setBusy(null)}
  }
  async function analyze(){
    if(!audio||!transcript.trim())return
    setError('');setAnalysis(null);setBusy('analyze')
    try{const form=new FormData();form.append('audio',audio);form.append('text',transcript.trim());const result=await apiData<Analysis>(await fetch(`${API}/analyze`,{method:'POST',body:form}));setAnalysis(result);setHistory(items=>[{id:Date.now(),text:transcript.trim(),sentiment:result.sentiment,score:result.sentiment_score,time:new Date().toLocaleTimeString()},...items].slice(0,8))}
    catch(e){setError(e instanceof Error?e.message:'Analysis failed')}finally{setBusy(null)}
  }
  async function startRecording(){
    setError('')
    try{
      const media=await navigator.mediaDevices.getUserMedia({audio:true});stream.current=media
      const mime=['audio/webm;codecs=opus','audio/webm','audio/mp4'].find(type=>MediaRecorder.isTypeSupported(type))
      const rec=new MediaRecorder(media,mime?{mimeType:mime}:undefined);recorder.current=rec;chunks.current=[]
      rec.ondataavailable=event=>{if(event.data.size)chunks.current.push(event.data)}
      rec.onstop=()=>{const blob=new Blob(chunks.current,{type:rec.mimeType||'audio/webm'});const ext=blob.type.includes('mp4')?'m4a':'webm';setAudio(new File([blob],`recording.${ext}`,{type:blob.type}));media.getTracks().forEach(track=>track.stop());stream.current=null;setRecording(false)}
      rec.start();setRecording(true)
    }catch(e){setError(e instanceof Error?e.message:'Microphone access is unavailable')}
  }
  function stopRecording(){recorder.current?.stop()}
  function onFile(file?:File){if(file){setAudio(file);setAnalysis(null);setError('')}}
  const score=analysis?.sentiment_score??0
  const meter=((Math.max(-3,Math.min(3,score))+3)/6)*100

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#top"><span className="brand-mark">M</span><span>Multimodal Sentiment Analysis<small>Speech + text understanding</small></span></a><div className="status"><i className={health?.model_ready?'online':''}/>{health?.model_ready?'Model ready':health?.status==='offline'?'API offline':'Model not trained'}</div></header>
    <main id="top">
      <section className="hero"><div><div className="eyebrow">MULTIMODAL SENTIMENT ANALYSIS</div><h1>Hear the words.<br/><em>Read the tone.</em></h1><p>Combine what is said with how it sounds to explore sentiment and emotion in a short speech clip.</p></div><div className="hero-art" aria-hidden="true"><div className="wave wave-one"/><div className="wave wave-two"/><div className="wave wave-three"/><div className="wave-core">♫</div></div></section>
      <div className="content-grid">
        <section className="panel input-panel"><div className="section-heading"><span className="step">01</span><div><h2>Speech and transcript</h2><p>Upload a clip or record one in your browser.</p></div></div>
          <div className="audio-actions"><button className={`record-button ${recording?'recording':''}`} onClick={recording?stopRecording:startRecording} type="button"><span className="record-dot"/>{recording?'Stop recording':'Record speech'}</button><button className="text-button" onClick={()=>input.current?.click()} type="button">Choose audio file</button><input ref={input} className="visually-hidden" type="file" accept="audio/*" onChange={e=>onFile(e.target.files?.[0])}/></div>
          <p className="hint">WAV, FLAC, MP3, M4A or browser recording · model window up to 12 seconds</p>
          {audio&&<div className="audio-preview"><div className="file-chip"><span>♫</span><div><strong>{audio.name}</strong><small>{(audio.size/1024/1024).toFixed(2)} MB</small></div><button aria-label="Remove audio" onClick={()=>{setAudio(null);setAnalysis(null)}}>×</button></div><audio controls src={audioUrl}/></div>}
          <div className="transcript-heading"><label htmlFor="transcript">Transcript</label><button className="transcribe-link" disabled={!audio||busy!==null} onClick={transcribe}>{busy==='transcribe'?'Transcribing…':'Transcribe locally ↗'}</button></div>
          <textarea id="transcript" value={transcript} onChange={e=>setTranscript(e.target.value)} placeholder="Type the words spoken, or transcribe the audio locally…" rows={6}/>
          <div className="privacy-note"><span>⌂</span><span>Audio is processed by your local backend and removed after the request.</span></div>
          {error&&<div className="error-box" role="alert">{error}</div>}
          <button className="primary-button" disabled={!audio||!transcript.trim()||busy!==null} onClick={analyze}>{busy==='analyze'?'Analyzing both signals…':'Analyze speech + text'}<span>→</span></button>
        </section>
        <section className="panel result-panel"><div className="section-heading"><span className="step">02</span><div><h2>Analysis</h2><p>Sentiment and emotion signals from the utterance.</p></div></div>
          {analysis?<div className="results"><div className="sentiment-result"><div className="result-label">PREDICTED SENTIMENT</div><div className={`sentiment-word ${analysis.sentiment}`}>{analysis.sentiment}</div><div className="score-line"><span>Negative</span><strong>{score.toFixed(2)} / 3</strong><span>Positive</span></div><div className="sentiment-track"><div className="track-center"/><div className="track-marker" style={{left:`${meter}%`}}/></div></div>
            <div className="result-divider"/><div className="emotion-head"><div><h3>Emotion signals</h3><p>Model scores for each emotion</p></div><span className="info-pill">MODEL OUTPUT</span></div>
            <div className="emotion-list">{Object.entries(analysis.emotions).sort((a,b)=>b[1]-a[1]).map(([name,value])=><div className="emotion-row" key={name}><div className="emotion-meta"><span>{name}</span><b>{Math.round(value*100)}%</b></div><div className="emotion-track"><div style={{width:`${Math.max(0,Math.min(1,value))*100}%`}}/></div></div>)}</div>
            {analysis.sarcasm_probability!==null&&<div className="sarcasm-pill">Sarcasm signal <strong>{Math.round(analysis.sarcasm_probability*100)}%</strong></div>}
            <p className="result-disclaimer">These are model predictions based on its training labels, not a definitive reading of a person’s feelings.</p>
          </div>:<div className="empty-results"><div className="empty-icon">◉</div><h3>Your results will appear here</h3><p>Add a recording and transcript, then run the analysis to see the fused prediction.</p><div className="modality-chips"><span>◉ Speech</span><span>▤ Text</span><span>✣ Fusion</span></div></div>}
        </section>
      </div>
      <section className="lower-grid"><div className="note-card"><span className="note-icon">✣</span><div><h3>How fusion works</h3><p>Separate speech and language encoders produce features. A learned gate combines them before the sentiment and emotion prediction heads.</p></div></div><div className="note-card caution"><span className="note-icon">!</span><div><h3>Read results with care</h3><p>The training corpus is scripted dialogue. Accents, noise, culture, and context affect performance; evaluate those limits before making broader claims.</p></div></div></section>
      {history.length>0&&<section className="history-section"><div className="history-title"><div><div className="eyebrow">THIS SESSION</div><h2>Recent analyses</h2></div><button className="text-button" onClick={()=>setHistory([])}>Clear history</button></div><div className="history-list">{history.map(item=><article key={item.id}><div className={`history-dot ${item.sentiment}`}/><div className="history-copy"><strong>{item.sentiment} · {item.score.toFixed(2)}</strong><p>{item.text}</p></div><time>{item.time}</time></article>)}</div></section>}
    </main><footer><span>Multimodal Sentiment Analysis · Final Year Project</span><span>Local processing · No account required</span></footer>
  </div>
}
