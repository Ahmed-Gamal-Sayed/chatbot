import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'


const Root = createRoot(document.getElementById('root'))
Root.render(
  <StrictMode>
    <App />
  </StrictMode>,
)
