import { useState, useCallback, useRef } from 'react';
import type { TraceOp } from '../types/telemetry';
import { uploadTrace, downloadLohalloc } from '../hooks/useApi';

export function TraceUpload(): JSX.Element {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ ops: number; bytes: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const text = await file.text();
      let trace: TraceOp[];

      if (file.name.endsWith('.json') || text.trim().startsWith('[')) {
        trace = JSON.parse(text) as TraceOp[];
      } else {
        const lines = text.trim().split('\n');
        trace = lines.slice(1).map((line) => {
          const [op, size, stackHash] = line.split(',').map((s) => s.trim());
          return { op: op as TraceOp['op'], size: Number(size), stack_hash: Number(stackHash) };
        });
      }

      const arrayBuffer = await uploadTrace(trace);
      downloadLohalloc(arrayBuffer, 'model.lohalloc');
      setResult({ ops: trace.length, bytes: arrayBuffer.byteLength });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="flex h-full flex-col p-4" data-testid="trace-upload">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Trace Upload</h3>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInputRef.current?.click()}
        className={`flex flex-1 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition ${
          dragOver ? 'border-cyan-400 bg-cyan-400/10' : 'border-slate-600 hover:border-slate-500'
        }`}
        data-testid="trace-dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.csv"
          className="hidden"
          onChange={(e) => { const file = e.target.files?.[0]; if (file) handleFile(file); }}
        />
        {uploading ? (
          <p className="text-sm text-slate-400">Replaying trace…</p>
        ) : (
          <>
            <p className="text-sm text-slate-400">
              Drag & drop a <code className="text-cyan-400">.json</code> or{' '}
              <code className="text-cyan-400">.csv</code> trace file
            </p>
            <p className="mt-1 text-xs text-slate-500">or click to browse</p>
          </>
        )}
      </div>
      {result && (
        <p className="mt-2 text-xs text-emerald-400" data-testid="upload-result">
          Replayed {result.ops} ops → {result.bytes} bytes downloaded
        </p>
      )}
      {error && <p className="mt-2 text-xs text-red-400" data-testid="upload-error">{error}</p>}
    </div>
  );
}