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
        // CSV schema: timestamp,op,size,stack_hash (header row skipped). The
        // timestamp (ns) is required — the server enforces it on replay.
        const lines = text.trim().split('\n');
        trace = lines.slice(1).map((line) => {
          const [timestamp, op, size, stackHash] = line
            .split(',')
            .map((s) => s.trim());
          return {
            timestamp: Number(timestamp),
            op: op as TraceOp['op'],
            size: Number(size),
            stack_hash: Number(stackHash),
          };
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
    [handleFile],
  );

  return (
    <div
      className="flex h-full flex-col bg-canvas text-ink font-mono"
      data-testid="trace-upload"
    >
      <div className="px-3 py-2 border-b border-ink-faint text-[10px] tracking-widest text-ink-muted">
        TRACE UPLOAD
     </div>
      <div
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInputRef.current?.click()}
        className={[
          'flex flex-1 cursor-pointer flex-col items-center justify-center',
          'border-2 border-dashed m-3 transition-colors duration-75',
          dragOver
            ? 'border-heat bg-heat/5'
            : 'border-ink-faint hover:border-ink-muted',
        ].join(' ')}
        data-testid="trace-dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        {uploading ? (
          <p className="text-[10px] text-ink-muted tracking-widest">
            REPLAYING TRACE...
        </p>
        ) : (
          <>
            <p className="text-[10px] text-ink-muted tracking-widest">
              DRAG & DROP <span className="text-ink">.JSON</span> OR{' '}
              <span className="text-ink">.CSV</span> TRACE
           </p>
            <p className="mt-1 text-[10px] text-ink-faint tracking-widest">
              OR CLICK TO BROWSE
         </p>
       </>
        )}
     </div>
      {result && (
        <p
          className="px-3 pb-2 text-[10px] text-heat tracking-widest"
          data-testid="upload-result"
        >
          REPLAYED {result.ops.toString().padStart(5, '0')} OPS {'->'} {result.bytes.toString().padStart(6, '0')} BYTES
       </p>
      )}
      {error && (
        <p
          className="px-3 pb-2 text-[10px] text-heat truncate"
          data-testid="upload-error"
        >
          ERR: {error}
       </p>
      )}
   </div>
  );
}