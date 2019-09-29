
import path = require("path");
import fs = require("fs");

class DevelopmentError extends Error { }

class Random {
  // XorShift
  x = 123456789
  y = 362436069
  z = 521288629
  w: number

  constructor(seed = 88675123) {
    this.w = seed
  }

  next(): number {
    let t = this.x ^ (this.x << 11)
    this.x = this.y
    this.y = this.z
    this.z = this.w
    return this.w = (this.w ^ (this.w >> 19)) ^ (t ^ (t >> 8));
  }

  range(min: number, max: number): number {
    return (this.next() % (max - min)) + min
  }

  shuffle<T>(array: Array<T>): Array<T> {
    const result = []
    const copy = array.concat([])
    for (let i = 0; i < array.length; i++) {
      const index = this.range(0, copy.length)
      result.push(copy[index])
      copy.splice(index, 1)
    }
    return result
  }
}
const random = new Random()

// Logger

const LOG_DIR_PATH = './log'
const LOG_DEBUG_PATH = LOG_DIR_PATH + '/debug'

type LogLevel = 'ERROR' | 'WARN' | 'DEBUG'

class Log {
  static debug(tag: any, text: any): void {
    Log._write(Log._format('DEBUG', tag, text))
  }

  static error(tag: any, text: any): void {
    Log._write(Log._format('ERROR', tag, text))
  }

  static warn(tag: any, text: any): void {
    Log._write(Log._format('WARN', tag, text))
  }

  static print(tag: any, text: any): void {
    console.log(Log._format('DEBUG', tag, text).trim())
  }

  static _format(level: LogLevel, tag: any, text: any): string {
    return `[${level}]: `.padStart(5 + 4) + `[${tag}]: `.padStart(16 + 4) + `${text}\n`
  }

  static _write(text: string): void {
    fs.appendFileSync(LOG_DEBUG_PATH, text)
  }

  static init() {
    fs.mkdirSync(path.dirname(LOG_DEBUG_PATH), { recursive: true })
    fs.writeFileSync(LOG_DEBUG_PATH, '')
  }
}
Log.init()

class CorrectCollector {
  static _corrects = new Array<boolean>()

  static push(correct: boolean): void {
    this._corrects.push(correct)
    if (20 < this._corrects.length) {
      this._corrects.shift()
    }
  }

  static rate(): number {
    const correct_size = this._corrects.reduce((total: number, correct: boolean) => correct ? total + 1 : total, 0)
    return correct_size / this._corrects.length
  }
}

const makeId = (function() {
  let count = 1
  return function makeId(): number {
    return count++
  }
}())

function setUnion<T>(a: Set<T>, b: Set<T>): Set<T> {
  return new Set([...a, ...b])
}

function setIntersection<T>(a: Set<T>, b: Set<T>): Set<T> {
  return new Set([...a].filter(x => b.has(x)))
}

function setDifference<T>(a: Set<T>, b: Set<T>): Set<T> {
  return new Set([...a].filter(x => !b.has(x)))
}

function never(): never {
  throw new DevelopmentError('Never')
}

namespace NN {
  const INPUT_SLOT_SIZE = 3
  // const MAX_DIGITS = 10
  // DEBUG: For ease of viewing
  const MAX_DIGITS = 3

  // Node
  const INITIAL_SIGNAL = 0
  const CORRECT_THRESHOLD_SIGNAL = 1
  const WRONG_THRESHOLD_SIGNAL = -1
  const POSITIVE_THRESHOLD_SIGNAL = 0.8
  const NEGATIVE_THRESHOLD_SIGNAL = -0.8

  // Edge
  const INITIAL_POSITIVE_WIGHT = 0.15
  const INITIAL_NEGATIVE_WIGHT = -0.15
  const DELETE_THRESHOULD_WEIGHT = 0.1
  const EDGE_TO_CORRECT_RATE = 1

  type Signal = number
  type Effect = number

  const enum NodeType { Input, Middle, Output }

  class Node {
    id: number
    type: NodeType
    signal: Signal = INITIAL_SIGNAL
    input = new Set<Edge>()
    output = new Set<Edge>()

    constructor(type: NodeType) {
      this.id = makeId()
      this.type = type
    }
  }

  class Edge {
    id: number
    weight: number
    source: Node
    destination: Node

    constructor(source: Node, destination: Node, weight: number) {
      this.id = makeId()
      this.weight = weight
      this.source = source
      this.destination = destination
    }
  }

  class AllNode {
    static _nodes = new Set<Node>()

    static add(node: Node): void {
      this._nodes.add(node)
    }

    static delete(node: Node): void {
      this._nodes.delete(node)
    }

    static size(): number {
      return this._nodes.size
    }

    static [Symbol.iterator](): Iterator<Node> {
      return this._nodes[Symbol.iterator]()
    }
  }

  type InputNodes = Node[]
  type OutputNodes = Node[]
  // type GraphElement = Edge | Node
  type Path = Array<Edge | Node>

  type InputData<Element> = Array<Element>

  enum LearnOutput { Hit, Nothing }
  type InputMap<Element> = Array<Map<Element, Node>>
  type OutputMap<Output> = Map<Output, Node>

  export function main() {
    const elements = ['h', 'i', 'e']

    const testData = makeTestData(elements, 10)

    const [inputMap] = makeInputMap(elements, INPUT_SLOT_SIZE)

    const [outputMap] = makeOutputMap([LearnOutput.Hit, LearnOutput.Nothing])

    for (let index in testData) {
      Log.debug('input count', index)
      Log.print('input count', index)
      let t = Date.now()
      applyNN(inputMap, outputMap, testData[index])
      Log.debug('time', Date.now() - t)
      Log.print('time', Date.now() - t)
    }
  }

  // No Genre

  function applyNN<Element, Output>(inputMap: InputMap<Element>, outputMap: OutputMap<Output>, input: InputData<Element>): void {
    applyInput(inputMap, input)
    const [answer] = decideAnswer(outputMap)
    const correct = getCorrect(input)
    Log.debug('input', input)
    Log.debug('correct', answer === correct)
    CorrectCollector.push(answer === correct)
    Log.debug('correct rate', CorrectCollector.rate())
    Log.print('correct rate', CorrectCollector.rate())
    Log.debug('isCorrect', answer === correct)
    Log.debug('AllNode.size', AllNode.size())
    showInputMap(inputMap)
    showOutputMap(outputMap)
    showAllNodes(inputMap)
    feedbackNN(outputMap, correct)
    resetNodes()
  }

  function feedbackNN<Output>(outputMap: OutputMap<Output>, correct: Output): void {
    // TODO: Consider whether to add randomness.
    // I think that adding randomness improves accuracy.
    // But reproducibility is lost.
    // Need to benchmark
    for (let [output, node] of outputMap) {
      const gap = measureGap(node, correct == output)
      feedbackEdge(node, gap)
    }

    for (let [output, node] of outputMap) {
      feedbackNode(node, correct == output)
    }
  }

  function feedbackNode(node: Node, is_correct: boolean): void {
    const { positiveNodes, negativeNodes } = extractBigSignalNodes()

    if (is_correct && 1 < positiveNodes.length) {
      let new_node = makeNode(NodeType.Middle)
      bind(new_node, node, INITIAL_POSITIVE_WIGHT)
      for (let n of positiveNodes) {
        bind(n, new_node, INITIAL_POSITIVE_WIGHT)
      }
    }

    if (!is_correct && 1 < negativeNodes.length) {
      let new_node = makeNode(NodeType.Middle)
      bind(new_node, node, INITIAL_NEGATIVE_WIGHT)
      for (let n of negativeNodes) {
        bind(n, new_node, INITIAL_NEGATIVE_WIGHT)
      }
    }
  }

  function extractBigSignalNodes(): { positiveNodes: Node[], negativeNodes: Node[] } {
    const positiveNodes = []
    const negativeNodes = []

    for (let node of AllNode) {
      if (node.type == NodeType.Output) continue

      if (POSITIVE_THRESHOLD_SIGNAL < node.signal) {
        positiveNodes.push(node)
      } else if (node.signal < NEGATIVE_THRESHOLD_SIGNAL) {
        negativeNodes.push(node)
      }
    }

    return { positiveNodes, negativeNodes }
  }

  function feedbackEdge(node: Node, gap: number): void {
    const depth = getBackDepth(node)
    // sum arithmetic sequence
    const total_depth = depth * (1 + depth) / 2

    const assignGap = gap * depth / total_depth
    const remainGap = gap - assignGap

    const total_weight = [...node.input].reduce((result: number, edge: Edge) => result + edge.weight, 0)

    for (let edge of node.input) {
      const weight_rate = edge.weight / total_weight
      edge.weight += assignGap * weight_rate
      edge.weight = roundNumber(edge.weight)
      if (Math.abs(edge.weight) < DELETE_THRESHOULD_WEIGHT) {
        deleteEdge(edge)
      }
      if (isNaN(edge.weight) || edge.weight === Infinity || edge.weight === -Infinity) {
        Log.error('edge.weight', edge.weight)
        Log.error('edge.id', edge.id)
        Log.error('depth', depth)
        Log.error('total_depth', total_depth)
        Log.error('gap', gap)
        Log.error('assignGap', assignGap)
        Log.error('total_weight', total_weight)
        Log.error('weight_rate', weight_rate)
        throw new DevelopmentError('edge.weight is NaN')
      }
      feedbackEdge(edge.source, remainGap)
    }
  }

  // function feedbackEdge(node: Node, gap: number): void {
  //   const paths = getBackPaths(node, [])
  //   const effectMap = getEdgeEffects(node)
  //   for (let [edge, effect] of effectMap) {
  //     edgeToCorrect(edge, gap, effect)
  //   }
  //
  //   for(let edge of node.input){
  //     feedbackEdge(edge.source, gap)
  //   }
  // }

  // function edgeToCorrect(edge: Edge, gap: number, edgeEffect: Effect): void {
  //   edge.weight = roundNumber(edge.weight * gap * edgeEffect * EDGE_TO_CORRECT_RATE, MAX_DIGITS)
  //
  //   if (Math.abs(edge.weight) < DELETE_THRESHOULD_WEIGHT) {
  //     deleteEdge(edge)
  //   }
  // }

  // function getEdgeEffects(node: Node): Map<Edge, Effect> {
  //   const result = new Map()
  //   const total_weight = sumEdgeEffect(node)
  //   const edges = backGenealogyEdges(node)
  //
  //   for(let edge of edges){
  //     result.set(edge, edge.weight / total_weight)
  //   }
  //
  //   return result
  // }

  // function sumEdgeEffect(node: Node): number {
  //   let total_weight = 0
  //   for(let edge of node.input){
  //     total_weight += Math.abs(edge.weight)
  //     total_weight += sumEdgeEffect(edge.source)
  //   }
  //   return total_weight
  // }

  // function getEdgeEffects(paths: Set<Path>): Map<Edge, Effect> {
  //   // make pathToSignal
  //   const pathToSignal = new Map<Path, Signal>()
  //   let total_signal = 0
  //
  //   for (let path of paths) {
  //     const path_signal = getPathSignal(path)
  //     pathToSignal.set(path, path_signal)
  //     total_signal += path_signal
  //   }
  //
  //   // Find all paths affected by edge
  //   const edgeToPaths = new Map<Edge, Array<Path>>()
  //
  //   for (let path of paths) {
  //     for (let graphElement of path) {
  //       if (graphElement instanceof Node) continue
  //
  //       let paths = edgeToPaths.get(graphElement) || []
  //       paths.push(path)
  //       edgeToPaths.set(graphElement, paths)
  //     }
  //   }
  //
  //   // EdgeEffect = affected signal / total signal / self edge weight
  //   const result = new Map()
  //
  //   for (let [edge, paths] of edgeToPaths) {
  //     // calculate affected signal
  //     const affected_signal = paths.reduce((total, path) => {
  //       const signal = pathToSignal.get(path)
  //       if (undefined === signal) throw new DevelopmentError()
  //       return total + signal
  //     }, 0)
  //
  //     let effect = roundNumber(affected_signal / total_signal / edge.weight, MAX_DIGITS)
  //     if(isNaN(effect) || effect === Infinity || effect === -Infinity){
  //       Log.error('edgeEffect', effect)
  //       Log.error('affected_signal', affected_signal)
  //       Log.error('total_signal', total_signal)
  //       Log.error('edge.weight', edge.weight)
  //       throw new DevelopmentError('effect is NaN')
  //     }
  //     result.set(edge, effect)
  //   }
  //
  //   return result
  // }

  // function getPathSignal(path: Path): Signal {
  //   return path.reduce((total_signal, graphElement) => {
  //     if (graphElement instanceof Edge) {
  //       return total_signal * graphElement.weight
  //     } else if (graphElement instanceof Node) {
  //       return total_signal * graphElement.signal
  //     } else {
  //       return never()
  //     }
  //   }, 1)
  // }

  // function getBackPaths(node: Node, path: Path): Set<Path> {
  //   if (node.input.size === 0) {
  //     path.push(node)
  //     return new Set([path])
  //   } else {
  //     let result = new Set<Path>()
  //     for (let edge of node.input) {
  //       const new_path = path.concat([])
  //       new_path.push(edge)
  //       result = setUnion(result, getBackPaths(edge.source, new_path))
  //     }
  //     return result
  //   }
  // }

  function getCorrect<Element>(input: Element[]): any {
    return /hi/.test(input.join('')) ? LearnOutput.Hit : LearnOutput.Nothing
  }

  function measureGap(node: Node, is_correct: boolean): number {
    if (is_correct) {
      return CORRECT_THRESHOLD_SIGNAL - node.signal
    } else {
      return WRONG_THRESHOLD_SIGNAL - node.signal
    }
  }

  function decideAnswer<Output>(outputMap: OutputMap<Output>): [Output, Node] {
    let max_signal = -Infinity
    let result_output = null
    let result_node = null

    // Eliminate randomness
    for (let [output, node] of outputMap) {
      if (max_signal < node.signal) {
        max_signal = node.signal
        result_output = output
        result_node = node
      }
    }

    if (result_output === null || result_node === null) {
      const [output, node] = outputMap.entries().next().value
      result_output = output
      result_node = node
    }

    return [result_output, result_node]
  }

  // Make Input Data

  function makeTestData<Element>(elements: Element[], count: number): InputData<Element>[] {
    let result: InputData<Element>[] = []
    const allPattern = makePatterns(elements)

    for (let i = 0; i < count; i++) {
      result = result.concat(random.shuffle(allPattern))
    }

    return result
  }

  function makePatterns<Element>(elements: Element[], length?: number): InputData<Element>[] {
    length = length || elements.length
    return recMakePatterns([], elements, length)
  }

  function recMakePatterns<Element>(pattern: Element[], origin: Element[], size: number): any {
    if (pattern.length >= size) {
      return [pattern]
    }

    return origin.flatMap((value: Element): Element[] => {
      return recMakePatterns([...pattern, value], origin, size)
    })
  }

  // Display

  function showInputMap<Element>(inputMap: InputMap<Element>): void {
    Log.debug('InputMap:Start', '')

    inputMap.forEach((map, index) => {
      for (let [element, node] of map) {
        Log.debug('InputMap:Node', `index: ${index}, element: ${element}, id: ${node.id}, signal: ${node.signal}`)
      }
    })

    Log.debug('InputMap:End', '')
  }

  function showOutputMap(outputMap: OutputMap<any>): void {
    Log.debug('OutputMap:Start', '')

    for (let [output, node] of outputMap) {
      Log.debug('OutputMap:Node', `output: ${LearnOutput[output]}, id: ${node.id}, signal: ${node.signal}`)
    }

    Log.debug('OutputMap:End', '')
  }

  function showAllNodes<Element>(inputMap: InputMap<Element>): void {
    Log.debug('NN:Start', '')

    for (let map of inputMap) {
      for (let [, node] of map) {
        recShowAllNodes(node, [])
      }
    }

    Log.debug('NN:End', '')
  }

  function recShowAllNodes(node: Node, path: Path): void {
    path.push(node)

    if (node.output.size === 0) {
      let text = ''
      for (let graphElement of path) {
        if (graphElement instanceof Node) {
          text += `[ ${graphElement.id} : ${graphElement.signal} ] `
        } else if (graphElement instanceof Edge) {
          text += `= ${graphElement.id} : ${graphElement.weight} = `
        } else { never() }
      }
      Log.debug('NN:Path', text)
    }

    for (let edge of node.output) {
      recShowAllNodes(edge.destination, [...path, edge])
    }
  }

  // Serial Input Data

  function applyInput<Element>(inputMap: InputMap<Element>, inputData: InputData<Element>): void {
    inputMap.forEach((map, index) => {
      for (let [element, node] of map) {
        inputData[index] === element ? addSignal(node, 1) : addSignal(node, -1)
      }
    })
  }

  function makeInputMap<Element>(elements: Element[], serialSize: number): [InputMap<Element>, InputNodes] {
    const inputMap = []
    const nodes = []

    for (let index = 0; index < serialSize; index++) {
      inputMap.push(new Map())
      for (let e of elements) {
        let node = makeNode(NodeType.Input)
        inputMap[index].set(e, node)
        nodes.push(node)
      }
    }

    return [inputMap, nodes]
  }

  function makeOutputMap<Output>(outputs: Output[]): [OutputMap<Output>, OutputNodes] {
    const map = new Map()
    for (let o of outputs) {
      map.set(o, makeNode(NodeType.Output))
    }

    const nodes = Array.from(map.values())

    return [map, nodes]
  }

  // Core NN

  function addSignal(node: Node, value: number): void {
    node.signal = roundNumber(node.signal + value)
    if (isNaN(node.signal) || node.signal === Infinity || node.signal === -Infinity) {
      Log.error('node.signal', node.signal)
      Log.error('node.id', node.id)
      Log.error('node.input.size', node.input.size)
      Log.error('value', value)
      throw new DevelopmentError('node.signal is NaN')
    }

    for (let edge of node.output) {
      let tmp = value * edge.weight / edge.destination.input.size
      if (isNaN(tmp) || tmp === Infinity || tmp === -Infinity) {
        Log.error('add value', tmp)
        Log.error('value', value)
        Log.error('edge.weight', edge.weight)
        Log.error('edge.destination.input.size', edge.destination.input.size)
        throw new DevelopmentError('add value is NaN')
      }
      addSignal(edge.destination, value * edge.weight / edge.destination.input.size)
    }
  }

  function resetNodes() {
    for (let node of AllNode) {
      node.signal = 0
    }
  }

  function bind(source: Node, destination: Node, weight: number): Edge {
    // TODO: If the Edge has already been created, do nothing
    const edge = makeEdge(source, destination, weight)
    source.output.add(edge)
    destination.input.add(edge)
    return edge
  }

  function makeEdge(source: Node, destination: Node, weight: number): Edge {
    return new Edge(source, destination, weight)
  }

  function deleteEdge(edge: Edge): void {
    edge.source.output.delete(edge)
    edge.destination.input.delete(edge)

    if (edge.source.output.size <= 0 && edge.source.type === NodeType.Middle) {
      deleteNode(edge.source)
    }

    if (edge.destination.input.size <= 0 && edge.destination.type === NodeType.Middle) {
      deleteNode(edge.destination)
    }
  }

  function makeNode(type: NodeType): Node {
    let node = new Node(type)
    AllNode.add(node)
    return node
  }

  function deleteNode(node: Node): void {
    if (node.type !== NodeType.Middle) throw new DevelopmentError(`node.type: ${node.type}, node: ${node}`)

    AllNode.delete(node)

    let edges = new Set<Edge>()
    edges = setUnion(edges, node.input)
    edges = setUnion(edges, node.output)

    node.input.clear()
    node.output.clear()

    for (let edge of edges) {
      deleteEdge(edge)
    }
  }

  function getBackDepth(node: Node): number {
    let result = 0
    recGetBackDepth(node, 0, depth => {
      if (result < depth) {
        result = depth
      }
    })
    return result
  }

  function recGetBackDepth(node: Node, depth: number, func: (depth: number) => void): void {
    if (node.input.size === 0) {
      func(depth)
      return
    }

    for (let edge of node.input) {
      recGetBackDepth(edge.source, depth++, func)
    }
  }

  function backGenealogyEdges(node: Node): Edge[] {
    let result: Edge[] = []
    for (let edge of node.input) {
      result.push(edge)
      result = result.concat(backGenealogyEdges(edge.source))
    }
    return result
  }

  // Utility

  function roundNumber(number: number, digits?: number) {
    let disits = 10 ** (digits || MAX_DIGITS)
    return Math.round(number * disits) / disits
  }
}

NN.main()
