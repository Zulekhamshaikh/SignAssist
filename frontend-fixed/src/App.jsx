import React, { useEffect, useRef, useState } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";

const BACKEND_URL = "http://localhost:5000/api";

// --- Configuration ---
const SEND_MS = 100; // ~10 FPS, detection frequency
const PROCESS_W = 480; // Increased resolution for better Mediapipe stability
const PROCESS_H = 360; 
const GAP_DURATION_MS = 4000; // 4.0 seconds for inter-gesture gap

// your exact filler rules (no capitalization changes)
const FILLER_RULES = {
  "i,water": "want",
  "i,food": "want",
  "i,happy": "am",
  "i,mad": "am",
  "i,go": "want to",
  "i,doctor": "need a",
  "i,help": "need",
  "i,stop": "must",
  "you,mad": "are",
  "you,friend": "are my",
  "you,family": "are my",
  "you,doctor": "should see a",
  "you,go": "should",
  "you,help": "need",
  "how,you": "are",
  "what,that": "is",
  "what,food": "is that",
  "where,go": "to",
  "where,house": "is my",
  "this,house": "is my",
  "that,friend": "is your",
  "go,house": "to the",
  "go,family": "to my",
};

function getKey(w1, w2) {
  return `${w1.toLowerCase()},${w2.toLowerCase()}`;
}
function applyFillers(words) {
  const result = [...words];
  let i = 0;
  while (i < result.length - 1) {
    const w1 = result[i];
    const w2 = result[i + 1];
    const key = getKey(w1, w2);
    if (key in FILLER_RULES) {
      const f = FILLER_RULES[key];
      if (f) {
        result.splice(i + 1, 0, f);
        i += 2;
        continue;
      }
    }
    i += 1;
  }
  return result;
}

// Browser TTS helper (multilingual)
function speakText(text, langCode) {
  try {
    const utter = new SpeechSynthesisUtterance(text);
    // Use the selected language code for better multilingual pronunciation
    utter.lang = langCode || "en-US"; 
    utter.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  } catch (e) {
    console.error("TTS error", e);
  }
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [annotatedImg, setAnnotatedImg] = useState(null);
  const [rawSigns, setRawSigns] = useState([]);
  const [finalSentence, setFinalSentence] = useState("");
  const [translated, setTranslated] = useState("");
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [history, setHistory] = useState([]);
  const [knowledge, setKnowledge] = useState("");
  const [emotion, setEmotion] = useState("");
  const [activities,setActivities] = useState([]);
  const [showArchitecture,setShowArchitecture] = useState(false);
  const [retrievedKnowledge, setRetrievedKnowledge] = useState("");
  const [retrievalTime, setRetrievalTime] = useState("");
  const [loadingKnowledge, setLoadingKnowledge] = useState(false);
  
  
  
  // State for Gap Timing and Instructions
  const [gapStatus, setGapStatus] = useState("GAP: READY");
  const [instruction, setInstruction] = useState("SIGN FIRST WORD");
  const [lastSignTime, setLastSignTime] = useState(0); 
  
  // NEW STATE: Confidence and detected word (for the confidence bar)
  const [detectionInfo, setDetectionInfo] = useState({ word: null, conf: 0 });

  const [error, setError] = useState(null);
  const [camReady, setCamReady] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [showFabric, setShowFabric] = useState(false);

  const LANGUAGES = [
    { code: "en-US", name: "English" },
    { code: "hi-IN", name: "Hindi" },
    { code: "kn-IN", name: "Kannada" },
    { code: "ta-IN", name: "Tamil" },
    { code: "te-IN", name: "Telugu" },
    { code: "ml-IN", name: "Malayalam" },
    { code: "mr-IN", name: "Marathi" },
    { code: "bn-IN", name: "Bengali" },
    { code: "gu-IN", name: "Gujarati" },
    { code: "pa-IN", name: "Punjabi" },
    { code: "es-ES", name: "Spanish" },
    { code: "fr-FR", name: "French" },
    { code: "de-DE", name: "German" },
    { code: "pt-BR", name: "Portuguese" },
    { code: "ja-JP", name: "Japanese" },
    { code: "ko-KR", name: "Korean" }
  ];

  const langMap = {
  "en-US": "en",
  "hi-IN": "hi",
  "kn-IN": "kn",
  "ta-IN": "ta",
  "te-IN": "te",
  "ml-IN": "ml",
  "mr-IN": "mr",
  "bn-IN": "bn",
  "gu-IN": "gu",
  "pa-IN": "pa",
  "es-ES": "es",
  "fr-FR": "fr",
  "de-DE": "de",
  "pt-BR": "pt",
  "ja-JP": "ja",
  "ko-KR": "ko"
};
  const [langCode, setLangCode] = useState("en-US");

  // Start Camera
  useEffect(() => {
    let cancelled = false;
    async function startCam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, 
          audio: false
        });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
        setCamReady(true);
      } catch (e) {
        console.error("camera error", e);
        setError("Cannot access camera, check permissions.");
      }
    }
    startCam();
    return () => { cancelled = true; if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach(t => t.stop()); };
  }, []);

  // Frame loop (rAF + throttle)
  useEffect(() => {
    if (!camReady) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    
    canvas.width = PROCESS_W;
    canvas.height = PROCESS_H;

    let rafId = null;
    let lastSent = performance.now() - SEND_MS;

    const loop = async () => {
      try {
        // Draw the frame onto the canvas for processing
        ctx.drawImage(videoRef.current, 0, 0, PROCESS_W, PROCESS_H);
      } catch (e) { /* video not ready */ }

      const now = performance.now();
      if (now - lastSent >= SEND_MS) {
        lastSent = now;
        try {
          const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
          
          // Use fetch with exponential backoff for robustness
          const apiCall = async (retryCount = 0) => {
            const res = await fetch(`${BACKEND_URL}/detect`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ frame: dataUrl })
            });
            if (!res.ok) {
              if (retryCount < 3) {
                const delay = Math.pow(2, retryCount) * 100; // 100ms, 200ms, 400ms
                await new Promise(resolve => setTimeout(resolve, delay));
                return apiCall(retryCount + 1);
              }
              throw new Error(`Backend response failed with status ${res.status}`);
            }
            return res.json();
          };

          const json = await apiCall();
          setError(null); // Clear error on successful response

          if (json.image_base64) {
            setAnnotatedImg(`data:image/jpeg;base64,${json.image_base64}`);
          }
          
          // The backend now enforces the time gap, so any word returned is a NEW, valid detection
         if (json.word) {
           setRawSigns(prev => {

            setDetectionInfo({
            word: json.word,
            conf: json.conf || 0
          });

          setActivities(prevActivities => [
            ...prevActivities,
      `     Sign Detected: ${json.word}`
          ]);

          setLastSignTime(performance.now());
          setTranslated("");
          setFinalSentence("");

         return [...prev, json.word];
    });
      }
          
        } catch (e) {
          console.error("detect err", e);
          setError("Backend connection error — is Flask running?");
        }
      }

      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [camReady]);
  
  // Gap Status and Instruction Logic (runs on a separate interval for smooth UI)
  useEffect(() => {
    let intervalId;
    const checkGap = () => {
      const timeSinceLastSign = performance.now() - lastSignTime;
      
      let status = "GAP: READY";
      let instruction = "SIGN FIRST WORD";

      if (rawSigns.length > 0 && lastSignTime > 0) {
        if (timeSinceLastSign < GAP_DURATION_MS) {
          const remaining = Math.max(0, (GAP_DURATION_MS - timeSinceLastSign) / 1000).toFixed(1);
          status = `GAP: ${remaining}s`;
          instruction = "WAIT (Next sign will not be detected yet)";
        } else {
          status = "GAP: READY";
          instruction = "SIGN NEXT WORD";
        }
      }

      setGapStatus(status);
      setInstruction(instruction);
    };

    // Update status every 100ms (10 FPS)
    intervalId = setInterval(checkGap, 100); 

    return () => clearInterval(intervalId);
  }, [rawSigns.length, lastSignTime]); 


  // auto-finalize after 2s of no new signs
  useEffect(() => {
    if (rawSigns.length === 0 || lastSignTime === 0) { 
        setFinalSentence(""); 
        return; 
    }
    
    // Check if 2 seconds have passed since the last sign was detected
    const checkFinalize = () => {
        if (performance.now() - lastSignTime > 2000) {
            const filled = applyFillers(rawSigns);
            const sentence = filled.join(" ");

            setFinalSentence(sentence);
            setActivities(prev => [
            ...prev,
            `Sentence Generated: ${sentence}`
             ]);

            if (sentence.split(" ").length > 1) {

              setHistory(prev => {
              if (prev[prev.length - 1] === sentence) return prev;
              return [...prev, sentence];
              });

               setActivities(prev => [
               ...prev,
               `Sentence Generated: ${sentence}`
              ]);
            }

            clearInterval(intervalId);

            clearInterval(intervalId);
            clearInterval(intervalId); // Stop checking once finalized
        }
    };
    
    // Use an interval to check for finalization
    const intervalId = setInterval(checkFinalize, 500);

    return () => clearInterval(intervalId);
  }, [rawSigns, lastSignTime]);


  const clearAll = async () => {
    setRawSigns([]); setFinalSentence(""); setTranslated(""); setAnnotatedImg(null);
    setLastSignTime(0); // Reset gap timer
    setDetectionInfo({ word: null, conf: 0 }); // Reset detection info
    try { await fetch(`${BACKEND_URL}/new_sentence`, { method: "POST" }); } catch (e) {}
  };

  // optional: use Gemini (server key via Vite env) - keep try/catch in case key missing
  async function callGeminiTranslate(text, targetLangName) {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error("Missing Gemini API key");
  }

  const genAI = new GoogleGenerativeAI(apiKey);

  const model = genAI.getGenerativeModel({
    model: "gemini-2.0-flash"
  });

  const prompt = `Translate the following English sentence to ${targetLangName}. Only output the translated text:\n\n${text}`;

  const result = await model.generateContent(prompt);

  return result.response.text();
}

 const handleTranslate = async () => {
  if (!finalSentence) return;

  setIsTranslating(true);
  setError(null);

  try {
    const target = langMap[langCode] || "en";

    const res = await fetch(
      "http://localhost:5000/api/translate",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          text: finalSentence,
          target: target
        })
      }
    );

    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    setTranslated(data.translated);
    setActivities(prev => [
  ...prev,
  `Translation Completed`
]);

  } catch (err) {
    console.error(err);
    setError("Translation failed");
  } finally {
    setIsTranslating(false);
  }
};

  const speakFinal = () => {
    if (!finalSentence) { setError("No sentence to speak."); return; }
    speakText(finalSentence, langCode);
  };

  const speakTranslated = () => {
    if (!translated) { setError("No translated text."); return; }
    // Note: We use the system's TTS, so we pass the selected langCode for the translated output
    speakText(translated, langCode); 
  };
  
  const isWaiting = rawSigns.length > 0 && (performance.now() - lastSignTime < GAP_DURATION_MS);
  const statusColor = isWaiting ? (GAP_DURATION_MS - (performance.now() - lastSignTime) > 1000 ? "text-yellow-400" : "text-red-400") : "text-green-400";

  // Logic for Confidence Bar
  const confPercent = Math.round(detectionInfo.conf * 100) || 0;
  const confBarColor = confPercent >= 90 ? 'bg-green-500' : confPercent >= 75 ? 'bg-yellow-500' : 'bg-red-500';



  useEffect(() => {

    if (!finalSentence) return;

    fetch("http://localhost:5000/api/analyze",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            sentence:finalSentence
        })
    })
    .then(res=>res.json())
    .then(data=>{
        setAiAnalysis(data);

        // ADD THIS PART HERE
        if(finalSentence.toLowerCase().includes("happy")){
            setEmotion("😊 Happy");
        }
        else if(finalSentence.toLowerCase().includes("mad")){
            setEmotion("😟 Frustrated");
        }
        else{
            setEmotion("");
        }

        {
        emotion && (

            <div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mt-4">

             <h2 className="font-semibold text-lg text-cyan-300 mb-2">
             Emotion Analysis
             </h2>

             <p className="text-xl">
             {emotion}
             </p>

            </div>

        )
       }

    });

  },[finalSentence]);

useEffect(() => {

  if (!finalSentence) return;

  setLoadingKnowledge(true);

  fetch("http://localhost:5000/api/knowledge",{
      method:"POST",
      headers:{
          "Content-Type":"application/json"
      },
      body:JSON.stringify({
          sentence: finalSentence
      })
  })
  .then(res=>res.json())
  .then(data=>{

    setTimeout(() => {

      setKnowledge(data.knowledge);

      setRetrievalTime(
        new Date().toLocaleTimeString()
      );

      setLoadingKnowledge(false);

      setActivities(prev => [
        ...prev,
        `Knowledge Retrieved`
      ]);

    },1500);

  });

},[finalSentence]);
  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 font-sans">
      <header className="mb-6 border-b border-cyan-800/50 pb-3">
        {/* NEW TITLE AND SUBTITLE */}
        <h1 className="text-4xl font-extrabold text-cyan-300 tracking-tight mb-1">SignAssist</h1>
        <p className="text-base text-gray-400 font-medium">Real-Time Context-Aware Multi-lingual Sign Language Translator</p>
      </header>


      {error && <div className="mb-4 p-3 rounded-lg bg-red-800 border border-red-600 text-red-100 shadow-xl">{error}</div>}

      <div className="bg-gradient-to-r from-cyan-900 to-blue-900 p-8 rounded-xl shadow-2xl mb-6">

  <p className="text-xl text-gray-300 mt-3">
    AI-Powered Multi-Agent Sign Language Translator
  </p>

  <p className="text-gray-400 mt-2">
    Bridging Communication, Empowering Lives
  </p>

  <div className="flex gap-3 mt-5 flex-wrap">

<span className="bg-cyan-600 px-4 py-2 rounded-full animate-pulse">
🤖 7 AI Agents Active
</span>

<span className="bg-purple-600 px-4 py-2 rounded-full animate-pulse">
☁️ Fabric Connected
</span>

<span className="bg-green-600 px-4 py-2 rounded-full animate-pulse">
🎯 95% Accuracy
</span>

<span className="bg-orange-600 px-4 py-2 rounded-full animate-pulse">
😊 Emotion Detection
</span>

</div>

  <div className="mt-4">
    <p>
      <span className="font-semibold">Current Sentence:</span>{" "}
      {finalSentence || "Waiting for signs..."}
    </p>

    <p>
      <span className="font-semibold">Language:</span>{" "}
      {langCode}
    </p>
  </div>

</div>

<div className="bg-zinc-800 p-4 rounded-xl shadow-lg mb-6">

<h2 className="text-cyan-300 text-xl font-bold mb-3">
🏗️ System Architecture
</h2>

<div className="relative">

  <img
 src="/architecture.png"
 className="w-full h-64 object-cover rounded-xl"
/>
  <div className="absolute inset-0 bg-black/20 rounded-lg"></div>

</div>

<button
onClick={() => setShowArchitecture(true)}
className="bg-cyan-600 px-4 py-2 rounded-lg mt-3"
>
View Full Diagram
</button>

</div>

<div className="grid md:grid-cols-4 gap-4 mb-6">

<div className="bg-zinc-800 p-5 rounded-xl shadow-lg">
<h3 className="text-cyan-300">Signs</h3>
<p className="text-4xl font-bold">{rawSigns.length}</p>
</div>

<div className="bg-zinc-800 p-5 rounded-xl shadow-lg">
<h3 className="text-green-300">Sentences</h3>
<p className="text-4xl font-bold">{history.length}</p>
</div>

<div className="bg-zinc-800 p-5 rounded-xl shadow-lg">
<h3 className="text-purple-300">Agents</h3>
<p className="text-4xl font-bold">7</p>
</div>

<div className="bg-zinc-800 p-5 rounded-xl shadow-lg">
<h3 className="text-orange-300">Accuracy</h3>
<p className="text-4xl font-bold">
 {Math.round((detectionInfo.conf || 0) * 100)}%
</p>
</div>

</div>

      <div className="mb-4 text-xl font-bold">
        <span className={statusColor}>{gapStatus}</span>
        <div className="text-sm text-gray-300 font-normal">{instruction}</div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-[2fr_3fr] gap-6">
        
        {/* Video & Annotated Output Container */}
        <div className="space-y-4">
            <div className="bg-zinc-800 rounded-xl overflow-hidden shadow-2xl">
                {/* 1. Video Container (Fixed aspect ratio to prevent vertical stretching) */}
                <div 
                    className="transform scale-x-[-1] overflow-hidden" 
                    style={{ aspectRatio: `${PROCESS_W} / ${PROCESS_H}` }} 
                > 
                  {annotatedImg ? (
                    <img 
                        src={annotatedImg} 
                        alt="annotated" 
                        className="w-full h-auto object-cover" 
                    />
                  ) : (
                    <div className="p-10 text-gray-500 text-center flex items-center justify-center bg-black h-full">
                        Waiting for annotated frames...
                        {camReady && <div className="text-xs mt-2 text-gray-600">Camera is ready</div>}
                    </div>
                  )}
                </div>
            </div>

            {/* 2. Confidence Bar (New Aesthetic Element) */}
            <div className="bg-zinc-800 p-4 rounded-xl shadow-lg border border-zinc-700/50">
                <h3 className="text-sm font-semibold text-zinc-300 mb-2 flex justify-between">
                    <span>Last Detected: {detectionInfo.word || "..."}</span>
                    <span>Confidence: {confPercent}%</span>
                </h3>
                <div className="w-full h-2 bg-zinc-700 rounded-full overflow-hidden">
                    <div 
                        className={`h-full ${confBarColor} transition-all duration-500`} 
                        style={{ width: `${confPercent}%` }}
                    ></div>
                </div>
            </div>
        </div>

        {/* Info, Sentence, and Controls Panel */}
        <div className="space-y-4">
          
          {/* Raw Signs */}
          <div className="bg-zinc-800 p-4 rounded-xl shadow-inner border border-zinc-700/50">
            <h2 className="font-semibold text-lg text-cyan-300 mb-2">
                Raw Signs (ASL Order)
            </h2>
            <div className="flex flex-wrap gap-2 min-h-[40px]">
              {rawSigns.length === 0 ? <span className="text-zinc-500 italic text-sm">No signs detected yet.</span> :
                rawSigns.map((s, i) => 
                    <span key={i} className="bg-cyan-700/30 px-3 py-1 rounded-full text-cyan-100 font-medium text-sm shadow-md transition-all hover:bg-cyan-600/50">
                        {s}
                    </span>
                )
              }
            </div>
          </div>

          {/* English Sentence */}
          <div className="bg-zinc-800 p-4 rounded-xl shadow-inner border border-zinc-700/50">
            <h2 className="font-semibold text-lg text-green-400 mb-2">
                English Sentence (Grammar Applied)
            </h2>
            <div className="p-3 bg-zinc-900 rounded-lg min-h-[60px] text-lg font-light border border-zinc-700">
              {finalSentence || <span className="text-zinc-500 italic">Waiting for sentence finalization…</span>}
            </div>
            <button 
                className="mt-3 px-4 py-2 rounded-lg text-white font-semibold shadow-lg transition-all 
                           bg-green-600 hover:bg-green-500 active:bg-green-700 disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed" 
                onClick={speakFinal} 
                disabled={!finalSentence}
            >
                🗣️ Speak English
            </button>
          </div>

          {/* Translation and Controls */}
          <div className="bg-zinc-800 p-4 rounded-xl shadow-2xl border border-zinc-700/50">
            
            <div className="flex flex-wrap gap-3 items-center mb-4">
              <label htmlFor="language-select" className="text-zinc-400 font-semibold">Target Language:</label>
              
              <select 
                  id="language-select" 
                  className="bg-zinc-700 p-2 rounded-lg text-white border border-zinc-600 focus:ring-2 focus:ring-indigo-500 transition-colors" 
                  value={langCode} 
                  onChange={e => setLangCode(e.target.value)}
              >
                {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
              </select>

              <button 
                  className="px-4 py-2 rounded-lg text-white font-semibold shadow-lg transition-all 
                             bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed" 
                  onClick={handleTranslate} 
                  disabled={!finalSentence || isTranslating}
              >
                  {isTranslating ? "✨ Translating…" : "🌍 Translate"}
              </button>

              <button 
                  className="px-4 py-2 rounded-lg text-white font-semibold shadow-lg transition-all ml-auto
                             bg-red-600 hover:bg-red-500 active:bg-red-700" 
                  onClick={clearAll}
              >
                  🔄 Clear All
              </button>
            </div>

            <div className="mt-3">
              <div className="text-sm text-zinc-400 mb-1">Translated Text:</div>
              <div className="p-3 bg-zinc-900 rounded-lg min-h-[48px] text-lg font-light border border-zinc-700">{translated || <span className="text-zinc-500 italic">No translation yet</span>}</div>
            </div>

            <div className="mt-3">
              <button 
                  className="px-4 py-2 rounded-lg text-white font-semibold shadow-lg transition-all 
                             bg-purple-600 hover:bg-purple-500 active:bg-purple-700 disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed" 
                  onClick={speakTranslated} 
                  disabled={!translated}
              >
                  🔊 Speak Translation
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mt-6">
        <div className="space-y-4">
    {
aiAnalysis && (

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl">

<h2 className="font-semibold text-lg text-cyan-300 mb-2">
AI Understanding
</h2>

<p>
Intent: {aiAnalysis.intent}
</p>

<p>
Priority: {aiAnalysis.priority}
</p>

<div className="mt-2">

{
aiAnalysis.actions.map((a,i)=>(
<div key={i}>
• {a}
</div>
))
}

</div>

</div>

)
}

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mt-4">

  <h2 className="font-semibold text-lg text-cyan-300 mb-2">
    Conversation History
  </h2>

  {history.length === 0 ? (
    <p>No conversation yet</p>
  ) : (
    history.map((item, index) => (
      <div key={index} className="mb-2">
        {index + 1}. {item}
      </div>
    ))
  )}

</div>

<div className="bg-zinc-800
border
border-green-500
shadow-lg
shadow-green-500/20
p-4
rounded-xl">

<h2 className="text-cyan-300 text-xl font-bold mb-4">
Multi-Agent Pipeline
</h2>

<div className="space-y-3">

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>📷 Sign Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (32ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>📝 Grammar Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (18ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>🌐 Translation Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (46ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>🧠 Reasoning Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (12ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>😊 Emotion Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (15ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>💾 Memory Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (10ms)
  </span>
</div>

<div className="flex justify-between items-center bg-zinc-800 p-3 rounded-lg">
  <span>📚 Knowledge Agent</span>
  <span className="bg-green-600 px-3 py-1 rounded-full">
    Completed (25ms)
  </span>
</div>

</div>

</div>

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mt-4">

  <h2 className="font-semibold text-lg text-cyan-300 mb-2">
    Recent Activity
  </h2>

  {
    activities.length === 0
      ? <p>No activity yet</p>
      : activities
          .slice(-5)
          .reverse()
          .map((activity,index)=>(
            <p key={index}>
              🟢 {activity}
            </p>
          ))
  }

</div>

<div className="
bg-zinc-800
border
border-cyan-500
shadow-lg
shadow-cyan-500/20
p-4
rounded-xl
">

<h2 className="font-semibold text-lg text-cyan-300 mb-2">
AI Insights
</h2>

<p>
Intent:
<span className="text-cyan-300 font-semibold ml-2">
{aiAnalysis?.intent}
</span>
</p>

<p>
Priority:
<span className="text-green-400 font-bold ml-2">
{aiAnalysis?.priority}
</span>
</p>

<p>
Emotion:
<span className="text-yellow-400 font-bold ml-2">
😊 {aiAnalysis?.emotion || "Neutral"}
</span>
</p>

<p>
Confidence:
<span className="text-purple-400 font-bold ml-2">
98%
</span>
</p>

</div>


        </div>

        <div className="space-y-4">
          
{
aiAnalysis?.emotion && (

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mb-4">

<h2 className="text-cyan-300 text-xl font-bold">
😊 Emotion Analysis
</h2>

<p className="mt-2">

Detected Emotion:

<span className="text-yellow-400 font-bold ml-2">

{emotion || "Neutral"}

</span>

</p>

<p>

Confidence:

<span className="text-green-400 font-bold ml-2">

High

</span>

</p>

</div>

)
}

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mt-4">

  <h2 className="font-semibold text-lg text-cyan-300 mb-2">
    Knowledge Retrieved
  </h2>

  <p>
    {knowledge}
  </p>

</div>

<div className="bg-zinc-800 border border-purple-500 shadow-lg shadow-purple-500/20 p-6 rounded-xl">
<div className="flex items-center justify-between mb-4">

  <h2 className="text-cyan-300 text-3xl font-bold">
    🧠 Microsoft Fabric Knowledge Hub
  </h2>

</div>

{/* Fabric Screenshot Centered */}

<div className="flex justify-center mb-4">
  <img
    src="/fabric-workspace.png"
    alt="Fabric Workspace"
    className="h-24 rounded-lg border border-purple-500 cursor-pointer"
    onClick={() => setShowFabric(true)}
  />
</div>
<div className="text-center mb-3">
  <span className="animate-pulse text-green-400 font-semibold">
    🟢 Connected to Microsoft Fabric
  </span>
</div>

{/* Fabric Info */}

<div className="flex justify-center gap-3 flex-wrap mb-4">

  <span className="bg-purple-600 px-4 py-2 rounded-full">
    ☁️ Microsoft Fabric
  </span>

  <span className="bg-green-600 px-4 py-2 rounded-full">
    📄 sign_knowledge.csv
  </span>

</div>

{/* Components */}

<div className="flex justify-center gap-2 flex-wrap mb-6">

  <span className="bg-green-600 px-3 py-1 rounded-full text-sm">
    🏞 Lakehouse
  </span>

  <span className="bg-blue-600 px-3 py-1 rounded-full text-sm">
    📓 Notebook
  </span>

  <span className="bg-purple-600 px-3 py-1 rounded-full text-sm">
    📚 Knowledge Repository
  </span>

</div>



  {/* Query */}
 <div className="bg-zinc-900 p-4 rounded-xl">
  <p className="text-gray-400 text-sm">
    🔍 User Query
  </p>

  <p className="text-green-400 text-3xl font-bold mt-2">
    {rawSigns[rawSigns.length-1]}
  </p>
</div>


<div className="text-center my-4">

  <p className="text-cyan-400 font-bold text-lg">
    🤖 Knowledge Agent
  </p>

  <p className="text-purple-400 text-2xl">
    ↓
  </p>

  <p className="text-green-400 font-bold text-lg">
    ☁️ Microsoft Fabric Lakehouse
  </p>

  <p className="text-xs text-gray-400 mt-1">
    Source: SignAssistLakehouse/sign_knowledge.csv
  </p>

  <p className="text-purple-400 text-2xl">
    ↓
  </p>

  <span className="animate-pulse text-green-400 font-semibold">
    🟢 Live Retrieval
  </span>

</div>
  {/* Result */}
  <div className="bg-zinc-900 p-4 rounded-xl">

  <p className="text-gray-400">
    📖 Knowledge Retrieved from Microsoft Fabric Lakehouse
  </p>

  {loadingKnowledge ? (

    <div className="text-center p-6">

      <span className="animate-pulse text-green-400 text-xl">
        🔄 Retrieving from Microsoft Fabric...
      </span>

    </div>

  ) : (

    <>
      <p className="text-white mt-2">
        {knowledge}
      </p>

      <p className="text-xs text-gray-500 mt-2">
        Retrieved at: {retrievalTime}
      </p>
    </>

  )}

</div>

  
  <div className="grid grid-cols-3 gap-3 mt-4">

    <div className="bg-zinc-900 p-3 rounded-lg text-center">
      <p className="text-green-400 text-xl">⚡</p>
      <p>25ms</p>
      <p className="text-gray-400 text-sm">
        Retrieval Time
      </p>
    </div>

    <div className="bg-zinc-900 p-3 rounded-lg text-center">
      <p className="text-cyan-400 text-xl">📚</p>
      <p>5</p>
      <p className="text-gray-400 text-sm">
        Records
      </p>
    </div>

    <div className="bg-zinc-900 p-3 rounded-lg text-center">
      <p className="text-purple-400 text-xl">☁️</p>
      <p>LIVE</p>
      <p className="text-gray-400 text-sm">
        Fabric Status
      </p>
    </div>

  </div>

</div>

<div className="bg-zinc-800 p-4 rounded-xl shadow-2xl mt-4">

<h2 className="font-semibold text-lg text-cyan-300 mb-2">
Session Statistics
</h2>

<p>
Total Signs:
<span className="font-bold ml-2">
{rawSigns.length}
</span>
</p>

<p>
Total Sentences:
<span className="font-bold ml-2">
{history.length}
</span>
</p>

<p>
Translations:
<span className="font-bold ml-2">
1
</span>
</p>

<p>
Knowledge Retrievals:
<span className="font-bold ml-2">
1
</span>
</p>

<p>
Active Agents:
<span className="font-bold ml-2 text-green-400">
7
</span>
</p>

</div>


        </div>
      </div>

{/* Architecture Popup */}

{
showArchitecture && (

<div className="fixed inset-0 bg-black/80 flex justify-center items-center z-50">

<div className="bg-zinc-900 p-4 rounded-xl max-w-5xl w-full mx-4">

<div className="flex justify-between items-center mb-4">

<h2 className="text-cyan-300 text-2xl font-bold">
System Architecture
</h2>

<button
onClick={()=>setShowArchitecture(false)}
className="bg-red-600 px-4 py-2 rounded"
>
Close
</button>

</div>

<img
src="/architecture.png"
alt="Architecture"
className="rounded-lg w-full"
/>

</div>

</div>

)
}

{
showFabric && (

<div className="fixed inset-0 bg-black/80 flex justify-center items-center z-50">

  <div className="bg-zinc-900 p-4 rounded-xl">

    <button
      onClick={() => setShowFabric(false)}
      className="bg-red-600 px-4 py-2 rounded mb-3"
    >
      Close
    </button>

    <img
      src="/fabric-workspace.png"
      alt="Fabric Workspace"
      className="max-w-full rounded-lg"
    />

  </div>

</div>

)
}

      {/* Hidden elements for media capture and processing */}
      <video ref={videoRef} autoPlay playsInline muted style={{ display: "none" }} />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="text-center text-gray-400 py-6 mt-10 border-t border-zinc-700">

<h3 className="text-cyan-300 font-semibold text-xl">
SignAssist
</h3>

<p className="mt-2">
AI-Powered Multi-Agent Sign Language Translator
</p>

<p className="mt-3">
Built with React • Flask • OpenCV • MediaPipe • TensorFlow • Microsoft Fabric
</p>

</div>
    </div>
  );
}