import React from "react";

export type Status =
  | { type: "idle" }
  | { type: "saving"; message: string }
  | { type: "success" }
  | { type: "error"; message: string };

interface Props {
  status: Status;
}

export default function StatusFeedback({ status }: Props) {
  if (status.type === "idle") return null;

  const styles = {
    saving: "border-blue-200 bg-blue-50 text-blue-700",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-red-200 bg-red-50 text-red-700",
  };

  return (
    <div className={`mt-3 rounded-md border px-3 py-2 text-xs ${styles[status.type]}`}>
      {status.type === "saving" && (
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 animate-spin rounded-full border-2 border-blue-200 border-t-blue-600" />
          {status.message}
        </div>
      )}
      {status.type === "success" && (
        <div className="flex items-center gap-2">
          <svg
            className="h-3.5 w-3.5 text-emerald-600 animate-save-check"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <span>已保存</span>
        </div>
      )}
      {status.type === "error" && (
        <div className="flex items-center gap-2">
          <svg className="h-3.5 w-3.5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          {status.message}
        </div>
      )}
    </div>
  );
}
