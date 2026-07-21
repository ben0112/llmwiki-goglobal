// 拖放收集器:把 DataTransfer 展开为文件列表,文件夹递归遍历子目录。
// relMap 记录每个文件相对拖入根的路径(含顶层文件夹名),供上传端保留目录结构。
// 不支持 webkitGetAsEntry 的环境退化为平铺文件列表(文件夹条目被忽略)。

export interface DroppedFiles {
  files: File[]
  relMap: Map<File, string>
}

type Entry = {
  isFile: boolean
  isDirectory: boolean
  fullPath: string
  file: (cb: (f: File) => void, err?: (e: unknown) => void) => void
  createReader: () => {
    readEntries: (cb: (entries: Entry[]) => void, err?: (e: unknown) => void) => void
  }
}

async function readAllEntries(reader: ReturnType<Entry['createReader']>): Promise<Entry[]> {
  // readEntries 每批最多返回 100 条,必须循环读到空批为止
  const all: Entry[] = []
  for (;;) {
    const batch = await new Promise<Entry[]>((resolve) => reader.readEntries(resolve, () => resolve([])))
    if (batch.length === 0) return all
    all.push(...batch)
  }
}

async function walk(entry: Entry, out: DroppedFiles): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File | null>((resolve) => entry.file(resolve, () => resolve(null)))
    if (file) {
      out.files.push(file)
      out.relMap.set(file, entry.fullPath.replace(/^\//, ''))
    }
    return
  }
  if (entry.isDirectory) {
    const children = await readAllEntries(entry.createReader())
    for (const child of children) await walk(child, out)
  }
}

export async function collectDroppedFiles(dt: DataTransfer): Promise<DroppedFiles> {
  const out: DroppedFiles = { files: [], relMap: new Map() }
  // webkitGetAsEntry 必须在 drop 事件处理器内同步取出,之后 items 即失效
  const entries: (Entry | null)[] = Array.from(dt.items ?? []).map((item) => {
    const getter = (item as DataTransferItem & { webkitGetAsEntry?: () => Entry | null }).webkitGetAsEntry
    return typeof getter === 'function' ? getter.call(item) : null
  })
  if (entries.some(Boolean)) {
    for (const entry of entries) {
      if (entry) await walk(entry, out)
    }
    return out
  }
  return { files: Array.from(dt.files), relMap: new Map() }
}
