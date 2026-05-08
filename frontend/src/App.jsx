import { useState, useRef, useEffect } from 'react'

const API_BASE_URL = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '')

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [temperature, setTemperature] = useState(0.6)
  const [maxTokens, setMaxTokens] = useState(120)
  const [apiStatus, setApiStatus] = useState('checking')
  const chatRef = useRef(null)

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    const controller = new AbortController()

    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
        setApiStatus(res.ok ? 'online' : 'offline')
      } catch {
        setApiStatus('offline')
      }
    }

    checkHealth()
    return () => controller.abort()
  }, [])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    const history = messages.slice(-10)
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          history,
          temperature,
          max_tokens: maxTokens
        })
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail = data?.detail || data?.error || `Request failed with status ${res.status}`
        throw new Error(detail)
      }

      const response = typeof data.response === 'string' ? data.response : 'No response text returned by API.'
      setMessages((prev) => [...prev, { role: 'assistant', content: response }])
      setApiStatus('online')
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: 'Error: ' + e.message }])
      setApiStatus('offline')
    }

    setLoading(false)
  }

  const examples = [
    'What is machine learning?',
    'Explain quantum computing',
    'Write a haiku about code'
  ]

  return (
    <div className="h-screen flex flex-col" style={{ background: '#f5f3ef', color: '#3d3a34' }}>
      {/* Header */}
      <div className="px-6 py-4 border-b flex items-center gap-3" style={{ borderColor: '#e5e1d8' }}>
        <h1 className="text-xl font-medium" style={{ color: '#2d2a24' }}>NanoChat</h1>
        <span className="text-xs px-2 py-1 rounded" style={{ background: '#e8e4dc', color: '#6b665a' }}>162M</span>
        <span className="text-xs px-2 py-1 rounded" style={{ background: '#e8e4dc', color: '#6b665a' }}>
          {apiStatus === 'online' ? 'API Online' : apiStatus === 'offline' ? 'API Offline' : 'API Checking'}
        </span>
      </div>

      {/* Disclaimer */}
      <div className="px-6 py-2 text-xs text-center" style={{ background: '#eee9df', color: '#8a8578', borderBottom: '1px solid #e5e1d8' }}>
        ⚠️ 162M parameter research model — responses may be inaccurate. Best for creative tasks &amp; simple questions.
      </div>

      {/* Chat */}
      <div ref={chatRef} className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center">
            <h2 className="text-xl mb-6" style={{ color: '#8a8578' }}>Start a conversation</h2>
            <div className="flex gap-2 flex-wrap justify-center max-w-md">
              {examples.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInput(ex)
                  }}
                  className="px-4 py-2 rounded-lg text-sm transition-all hover:bg-opacity-80"
                  style={{
                    background: '#e8e4dc',
                    border: '1px solid #d1ccc3',
                    color: '#5a5549'
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className={`max-w-2xl ${msg.role === 'user' ? 'ml-auto' : 'mr-auto'}`}
            >
              <div
                className="px-4 py-3 rounded-2xl"
                style={{
                  background: msg.role === 'user' ? '#e8e4dc' : '#fff',
                  borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                  color: '#3d3a34',
                  border: msg.role === 'assistant' ? '1px solid #e5e1d8' : 'none'
                }}
              >
                {msg.content}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="max-w-2xl mr-auto">
            <div
              className="px-4 py-3 rounded-2xl"
              style={{ background: '#fff', border: '1px solid #e5e1d8', borderRadius: '16px 16px 16px 4px', color: '#8a8578' }}
            >
              thinking... longer replies are slower on this model
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t" style={{ borderColor: '#e5e1d8' }}>
        <div className="max-w-2xl mx-auto">
          <div className="flex gap-3 rounded-xl p-1" style={{ background: '#fff', border: '1px solid #e5e1d8' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type a message..."
              className="flex-1 bg-transparent px-4 py-3 text-sm outline-none"
              style={{ color: '#3d3a34' }}
            />
            <button
              onClick={sendMessage}
              disabled={loading}
              className="px-5 py-2 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: loading ? '#e5e1d8' : '#3d3a34',
                color: loading ? '#8a8578' : '#f5f3ef'
              }}
            >
              Send
            </button>
          </div>

          {/* Settings */}
          <div className="flex gap-6 mt-3 justify-center text-xs" style={{ color: '#8a8578' }}>
            <div className="flex items-center gap-2">
              <label>Temp:</label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-20"
                style={{ accentColor: '#8a8578' }}
              />
              <span className="w-6">{temperature}</span>
            </div>
            <div className="flex items-center gap-2">
              <label>Max:</label>
              <input
                type="range"
                min="10"
                max="500"
                step="10"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                className="w-20"
                style={{ accentColor: '#8a8578' }}
              />
              <span className="w-6">{maxTokens}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
