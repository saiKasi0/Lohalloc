import { useTelemetry } from './hooks/useTelemetry';
import { HeapMap } from './components/HeapMap';
import { PolicyMatrix } from './components/PolicyMatrix';
import { PerfTraceView } from './components/PerfTraceView';
import { StrategyToggle } from './components/StrategyToggle';
import { TraceUpload } from './components/TraceUpload';

function App() {
  const { records, isConnected } = useTelemetry();

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <header className="flex items-center justify-between border-b border-slate-700 px-6 py-3">
        <h1 className="text-xl font-bold text-cyan-400">Lohalloc Control Plane</h1>
        <div className="flex items-center gap-2 text-sm">
          <span
            className={`inline-block h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`}
          />
          <span className="text-slate-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
          <span className="text-slate-500">· {records.length} records</span>
        </div>
      </header>
      <main className="grid grid-cols-12 gap-3 p-3" style={{ height: 'calc(100vh - 56px)' }}>
        <div className="col-span-6 row-span-2 rounded-lg border border-slate-700 bg-slate-800">
          <HeapMap records={records} />
        </div>
        <div className="col-span-3 rounded-lg border border-slate-700 bg-slate-800">
          <PolicyMatrix records={records} />
        </div>
        <div className="col-span-3 row-span-2 rounded-lg border border-slate-700 bg-slate-800">
          <StrategyToggle />
        </div>
        <div className="col-span-3 rounded-lg border border-slate-700 bg-slate-800">
          <TraceUpload />
        </div>
        <div className="col-span-6 rounded-lg border border-slate-700 bg-slate-800">
          <PerfTraceView records={records} />
        </div>
        <div className="col-span-3 rounded-lg border border-slate-700 bg-slate-800">
          <div className="p-4 text-sm text-slate-400">
            <h3 className="mb-2 font-semibold text-slate-200">Telemetry Stats</h3>
            <p>Records: {records.length}</p>
            <p>Allocs: {records.filter((r) => r.op === 'alloc').length}</p>
            <p>Frees: {records.filter((r) => r.op === 'free').length}</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;