import { BaseSolver } from "lib/solvers/BaseSolver/BaseSolver"
import type { InputProblem } from "lib/types/InputProblem"
import type { SolvedTracePath } from "lib/solvers/SchematicTraceLinesSolver/SchematicTraceLinesSolver"
import type { MspConnectionPairId } from "lib/solvers/MspConnectionPairSolver/MspConnectionPairSolver"
import { rectIntersectsAnyTrace } from "lib/solvers/NetLabelPlacementSolver/SingleNetLabelPlacementSolver/collisions"
import type { GraphicsObject } from "graphics-debug"
import { visualizeInputProblem } from "lib/solvers/SchematicTracePipelineSolver/visualizeInputProblem"
import type { Point } from "@tscircuit/math-utils"
import { generateElbowVariants } from "lib/solvers/SchematicTraceLinesSolver/SchematicTraceSingleLineSolver/generateElbowVariants"

type PendingNetLabel = {
  bounds: { minX: number; minY: number; maxX: number; maxY: number }
  globalConnNetId: string
}

/**
 * This solver nudges trace segments out of the way to reduce collisions with
 * already-determined netlabel rectangles. It operates conservatively:
 *  - Only adjusts the specific colliding segment of the specific path
 *  - Uses a small perpendicular jog if the segment is terminal, otherwise
 *    tries shifting the interior segment minimally to clear the rectangle
 */
export class NetLabelClearanceShiftSolver extends BaseSolver {
  inputProblem: InputProblem
  inputTracePaths: Array<SolvedTracePath>
  pendingNetLabels: Array<PendingNetLabel>

  correctedTraceMap: Record<MspConnectionPairId, SolvedTracePath> = {}

  // bookkeeping for iteration
  private queuedCollisions: Array<{
    label: PendingNetLabel
    mspPairId: MspConnectionPairId
    segIndex: number
  }> = []

  constructor(params: {
    inputProblem: InputProblem
    inputTracePaths: Array<SolvedTracePath>
    pendingNetLabels: Array<PendingNetLabel>
  }) {
    super()
    this.inputProblem = params.inputProblem
    this.inputTracePaths = params.inputTracePaths
    this.pendingNetLabels = params.pendingNetLabels

    for (const trace of this.inputTracePaths) {
      this.correctedTraceMap[trace.mspPairId] = trace
    }

    this.computeQueuedCollisions()
  }

  override getConstructorParams(): ConstructorParameters<
    typeof NetLabelClearanceShiftSolver
  >[0] {
    return {
      inputProblem: this.inputProblem,
      inputTracePaths: this.inputTracePaths,
      pendingNetLabels: this.pendingNetLabels,
    }
  }

  private computeQueuedCollisions() {
    this.queuedCollisions = []
    for (const label of this.pendingNetLabels) {
      const result = rectIntersectsAnyTrace(
        label.bounds,
        this.correctedTraceMap as any,
      )
      if (result.hasIntersection) {
        this.queuedCollisions.push({
          label,
          mspPairId: result.mspPairId as MspConnectionPairId,
          segIndex: (result as any).segIndex as number,
        })
      }
    }
  }

  private clonePoints(pts: Point[]): Point[] {
    return pts.map((p) => ({ x: p.x, y: p.y }))
  }

  private tryPerpendicularNudge(
    pts: Point[],
    si: number,
    label: PendingNetLabel,
  ): Point[] | null {
    const EPS = 1e-6
    const start = pts[si]!
    const end = pts[si + 1]!
    const isVert = Math.abs(start.x - end.x) < EPS
    const isHorz = Math.abs(start.y - end.y) < EPS
    if (!isVert && !isHorz) return null

    // Determine perpendicular offset direction away from the label
    const offsetMag = 0.5
    const offset = { x: 0, y: 0 }
    if (isVert) {
      const left = Math.min(label.bounds.minX, label.bounds.maxX)
      const right = Math.max(label.bounds.minX, label.bounds.maxX)
      // Push left if we're to the right of label, else push right
      if (start.x >= right + EPS) offset.x = offsetMag
      else if (start.x <= left - EPS) offset.x = -offsetMag
      else offset.x = start.x >= (left + right) / 2 ? offsetMag : -offsetMag
    } else {
      const bottom = Math.min(label.bounds.minY, label.bounds.maxY)
      const top = Math.max(label.bounds.minY, label.bounds.maxY)
      if (start.y >= top + EPS) offset.y = offsetMag
      else if (start.y <= bottom - EPS) offset.y = -offsetMag
      else offset.y = start.y >= (bottom + top) / 2 ? offsetMag : -offsetMag
    }

    const newPts = this.clonePoints(pts)

    // If terminal segment, insert a short jog; else slide the two vertices
    const isTerminal = si === 0 || si === pts.length - 2
    if (isTerminal) {
      // Create a small dogleg jog perpendicular to current segment
      const J = 0.6
      if (isVert) {
        const jogY = si === 0 ? start.y + Math.sign(end.y - start.y) * J : end.y - Math.sign(end.y - start.y) * J
        if (si === 0) {
          newPts.splice(
            1,
            1,
            { x: start.x, y: jogY },
            { x: start.x + offset.x, y: jogY },
            { x: end.x + offset.x, y: end.y },
          )
        } else {
          newPts.splice(
            si,
            1,
            { x: start.x + offset.x, y: start.y },
            { x: end.x + offset.x, y: jogY },
            { x: end.x, y: jogY },
          )
        }
      } else {
        const jogX = si === 0 ? start.x + Math.sign(end.x - start.x) * J : end.x - Math.sign(end.x - start.x) * J
        if (si === 0) {
          newPts.splice(
            1,
            1,
            { x: jogX, y: start.y },
            { x: jogX, y: start.y + offset.y },
            { x: end.x, y: end.y + offset.y },
          )
        } else {
          newPts.splice(
            si,
            1,
            { x: start.x, y: start.y + offset.y },
            { x: jogX, y: end.y + offset.y },
            { x: jogX, y: end.y },
          )
        }
      }
    } else {
      // Slide the shared vertices of the segment perpendicular to its axis
      if (isVert) {
        newPts[si] = { x: newPts[si]!.x + offset.x, y: newPts[si]!.y }
        newPts[si + 1] = { x: newPts[si + 1]!.x + offset.x, y: newPts[si + 1]!.y }
      } else {
        newPts[si] = { x: newPts[si]!.x, y: newPts[si]!.y + offset.y }
        newPts[si + 1] = { x: newPts[si + 1]!.x, y: newPts[si + 1]!.y + offset.y }
      }
    }

    return newPts
  }

  private l1PathLength(pts: Point[]): number {
    let sum = 0
    for (let i = 0; i < pts.length - 1; i++) {
      sum += Math.abs(pts[i + 1]!.x - pts[i]!.x) + Math.abs(pts[i + 1]!.y - pts[i]!.y)
    }
    return sum
  }

  private sumVertexDelta(a: Point[], b: Point[]): number {
    const n = Math.min(a.length, b.length)
    let sum = 0
    for (let i = 0; i < n; i++) {
      sum += Math.abs(a[i]!.x - b[i]!.x) + Math.abs(a[i]!.y - b[i]!.y)
    }
    return sum + Math.abs(a.length - b.length) // penalize inserted points
  }

  private tryElbowVariants(
    pts: Point[],
    mspPairId: MspConnectionPairId,
    label: PendingNetLabel,
  ): Point[] | null {
    const { elbowVariants } = generateElbowVariants({ baseElbow: pts, guidelines: [], maxVariants: 500 })
    const baseLen = this.l1PathLength(pts)
    let best: { pts: Point[]; collisions: number; len: number; delta: number } | null = null

    const countCollisionsFor = (variantPts: Point[]) => {
      const temp: Record<MspConnectionPairId, SolvedTracePath> = { ...(this.correctedTraceMap as any) }
      temp[mspPairId] = { ...this.correctedTraceMap[mspPairId]!, tracePath: variantPts }
      let c = 0
      for (const L of this.pendingNetLabels) {
        const r = rectIntersectsAnyTrace(L.bounds, temp, mspPairId)
        if (r.hasIntersection) c++
      }
      return c
    }

    for (const v of elbowVariants) {
      const collisions = countCollisionsFor(v)
      const len = this.l1PathLength(v)
      const delta = this.sumVertexDelta(pts, v)
      const candidate = { pts: v, collisions, len, delta }
      if (!best) best = candidate
      else {
        if (candidate.collisions < best.collisions - 1e-9) best = candidate
        else if (candidate.collisions === best.collisions) {
          if (candidate.delta < best.delta - 1e-9) best = candidate
          else if (Math.abs(candidate.delta - best.delta) < 1e-9 && candidate.len < best.len - 1e-9) best = candidate
        }
      }
      if (best && best.collisions === 0) {
        // optimal
        break
      }
    }

    if (best && best.collisions < countCollisionsFor(pts)) {
      // Ensure at least the targeted label is cleared
      const clearedTarget = this.clearsLabel(best.pts, label, mspPairId)
      if (clearedTarget) return best.pts
    }
    return null
  }

  private clearsLabel(
    pts: Point[],
    label: PendingNetLabel,
    selfId: MspConnectionPairId,
  ): boolean {
    const temp: Record<MspConnectionPairId, SolvedTracePath> = {
      ...(this.correctedTraceMap as any),
    }
    // temporarily substitute
    const anyId = selfId
    temp[anyId] = {
      ...this.correctedTraceMap[anyId]!,
      tracePath: pts,
    }
    const res = rectIntersectsAnyTrace(label.bounds, temp, anyId)
    return !res.hasIntersection
  }

  override _step() {
    if (this.queuedCollisions.length === 0) {
      this.solved = true
      return
    }

    const next = this.queuedCollisions.shift()!
    const { label, mspPairId, segIndex } = next
    const original = this.correctedTraceMap[mspPairId]!

    // First, try elbow variants to clear the label aesthetically
    let candidate = this.tryElbowVariants(original.tracePath, mspPairId, label)
    if (!candidate) {
      // Fallback to a small perpendicular jog/slide
      candidate = this.tryPerpendicularNudge(original.tracePath, segIndex, label)
    }

    if (candidate && this.clearsLabel(candidate, label, mspPairId)) {
      this.correctedTraceMap[mspPairId] = {
        ...original,
        tracePath: candidate,
      }
      // recompute collisions as geometry changed
      this.computeQueuedCollisions()
      return
    }

    // If unable to resolve, keep original and continue to next
    if (this.queuedCollisions.length === 0) {
      this.solved = true
    }
  }

  override visualize(): GraphicsObject {
    const g = visualizeInputProblem(this.inputProblem)
    for (const trace of Object.values(this.correctedTraceMap)) {
      g.lines!.push({ points: trace.tracePath, strokeColor: "#6a5acd" })
    }
    for (const label of this.pendingNetLabels) {
      const w = label.bounds.maxX - label.bounds.minX
      const h = label.bounds.maxY - label.bounds.minY
      g.rects!.push({
        center: {
          x: (label.bounds.minX + label.bounds.maxX) / 2,
          y: (label.bounds.minY + label.bounds.maxY) / 2,
        },
        width: w,
        height: h,
        fill: "rgba(255,165,0,0.15)",
        strokeColor: "rgba(255,165,0,0.8)",
      } as any)
    }
    return g
  }
}

