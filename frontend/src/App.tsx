import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Trading from './pages/Trading'
import ML from './pages/ML'
import Backtest from './pages/Backtest'

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      <nav className="flex items-center gap-4 px-6 py-3 bg-slate-800 border-b border-slate-700">
        <span className="font-extrabold text-lg text-blue-400 tracking-tight">IH</span>
        <NavLink to="/" end className={({ isActive }) =>
          `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-300 hover:bg-slate-700'
          }`
        }>Dashboard</NavLink>
        <NavLink to="/trading" className={({ isActive }) =>
          `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-300 hover:bg-slate-700'
          }`
        }>Trading</NavLink>
        <NavLink to="/ml" className={({ isActive }) =>
          `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-300 hover:bg-slate-700'
          }`
        }>ML</NavLink>
        <NavLink to="/backtest" className={({ isActive }) =>
          `px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-300 hover:bg-slate-700'
          }`
        }>Backtest</NavLink>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/trading" element={<Trading />} />
        <Route path="/ml" element={<ML />} />
        <Route path="/backtest" element={<Backtest />} />
      </Routes>
    </Layout>
  )
}
