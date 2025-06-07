"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  Video,
  Tag,
  Sparkles,
  Download,
  Play,
  ImageIcon,
  FileText,
  Database,
  Zap,
  Brain,
  Eye,
} from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface Product {
  type: string
  color: string
  match_type: "exact" | "similar" | "no_match"
  matched_product_id: string
  confidence: number
}

interface VideoResult {
  video_id: string
  vibes: string[]
  products: Product[]
}

interface BatchResult {
  total_videos: number
  processed_videos: number
  results: VideoResult[]
  processing_time: number
}

export default function FlickdFuturisticUI() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [caption, setCaption] = useState("")
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<VideoResult | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResult | null>(null)
  const [activeTab, setActiveTab] = useState("single")
  const [currentStep, setCurrentStep] = useState("")
  const [glowEffect, setGlowEffect] = useState(false)

  // Animated background particles
  useEffect(() => {
    const interval = setInterval(() => {
      setGlowEffect((prev) => !prev)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith("video/")) {
      setSelectedFile(file)
    }
  }

  const processSingleVideo = async () => {
    if (!selectedFile) return

    setProcessing(true)
    setProgress(0)

    const steps = [
      { name: "Extracting frames...", icon: "🎞️" },
      { name: "Running YOLOv8 detection...", icon: "🤖" },
      { name: "Generating CLIP embeddings...", icon: "⚡" },
      { name: "Matching with FAISS index...", icon: "🎯" },
      { name: "Classifying vibes with NLP...", icon: "🌟" },
      { name: "Generating final output...", icon: "🚀" },
    ]

    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(steps[i].name)
      await new Promise((resolve) => setTimeout(resolve, 1000))
      setProgress(((i + 1) / steps.length) * 100)
    }

    const captionLower = caption.toLowerCase()
    let vibes: string[] = []

    if (captionLower.includes("glam") || captionLower.includes("party") || captionLower.includes("sparkle")) {
      vibes.push("Party Glam")
    }
    if (captionLower.includes("bow") || captionLower.includes("pink") || captionLower.includes("feminine")) {
      vibes.push("Coquette")
    }
    if (captionLower.includes("minimal") || captionLower.includes("clean") || captionLower.includes("natural")) {
      vibes.push("Clean Girl")
    }
    if (captionLower.includes("street") || captionLower.includes("urban") || captionLower.includes("edgy")) {
      vibes.push("Streetcore")
    }
    if (captionLower.includes("cottage") || captionLower.includes("floral") || captionLower.includes("vintage")) {
      vibes.push("Cottagecore")
    }
    if (captionLower.includes("y2k") || captionLower.includes("2000") || captionLower.includes("metallic")) {
      vibes.push("Y2K")
    }
    if (captionLower.includes("boho") || captionLower.includes("bohemian")) {
      vibes.push("Boho")
    }

    if (vibes.length === 0) {
      vibes = ["Clean Girl"]
    }

    const mockResult: VideoResult = {
      video_id: `video_${Date.now().toString().slice(-6)}`,
      vibes: vibes.slice(0, 2),
      products: [
        {
          type: "dress",
          color: "black",
          match_type: "similar",
          matched_product_id: "prod_456",
          confidence: 0.84,
        },
        {
          type: "earrings",
          color: "gold",
          match_type: "exact",
          matched_product_id: "prod_789",
          confidence: 0.92,
        },
      ],
    }

    setResult(mockResult)
    setProcessing(false)
    setCurrentStep("")
  }

  const processBatchVideos = async () => {
    setProcessing(true)
    setProgress(0)

    const totalVideos = 10
    const results: VideoResult[] = []

    for (let i = 0; i < totalVideos; i++) {
      setCurrentStep(`🎬 Processing video ${i + 1}/${totalVideos}...`)
      await new Promise((resolve) => setTimeout(resolve, 600))
      setProgress(((i + 1) / totalVideos) * 100)

      const vibes = [
        ["Coquette", "Party Glam"],
        ["Clean Girl"],
        ["Streetcore", "Y2K"],
        ["Cottagecore"],
        ["Boho", "Clean Girl"],
        ["Party Glam"],
        ["Y2K", "Streetcore"],
        ["Coquette"],
        ["Clean Girl", "Cottagecore"],
        ["Party Glam", "Coquette"],
      ]

      const mockResult: VideoResult = {
        video_id: `video_${(i + 1).toString().padStart(3, "0")}`,
        vibes: vibes[i],
        products: [
          {
            type: ["dress", "top", "bag", "earrings", "shoes"][Math.floor(Math.random() * 5)],
            color: ["black", "white", "gold", "brown", "silver"][Math.floor(Math.random() * 5)],
            match_type: Math.random() > 0.3 ? "similar" : "exact",
            matched_product_id: `prod_${Math.floor(Math.random() * 900) + 100}`,
            confidence: Math.round((Math.random() * 0.3 + 0.7) * 100) / 100,
          },
        ],
      }

      results.push(mockResult)
    }

    setBatchResults({
      total_videos: totalVideos,
      processed_videos: totalVideos,
      results: results,
      processing_time: 45.2,
    })

    setProcessing(false)
    setCurrentStep("")
  }

  const downloadResult = (data: any, filename: string) => {
    const dataStr = JSON.stringify(data, null, 2)
    const dataBlob = new Blob([dataStr], { type: "application/json" })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement("a")
    link.href = url
    link.download = filename
    link.click()
    URL.revokeObjectURL(url)
  }

  const downloadAllResults = () => {
    if (!batchResults) return
    batchResults.results.forEach((result) => {
      downloadResult(result, `${result.video_id}_result.json`)
    })
    downloadResult(batchResults, "batch_processing_summary.json")
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-purple-900/20 via-slate-900 to-black"></div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className={`absolute w-1 h-1 bg-cyan-400 rounded-full animate-pulse ${glowEffect ? "opacity-100" : "opacity-30"} transition-opacity duration-2000`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 2}s`,
              animationDuration: `${2 + Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.03)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none"></div>

      <div className="relative z-10 max-w-7xl mx-auto p-6 space-y-8">
        {/* Futuristic Header */}
        <div className="text-center space-y-6">
          <div className="flex items-center justify-center gap-4 mb-8">
            <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
              FLICKD.AI
            </h1>
          </div>

          <div className="backdrop-blur-sm bg-white/5 border border-cyan-500/20 rounded-2xl p-6 shadow-2xl">
            <p className="text-xl text-cyan-100 max-w-3xl mx-auto mb-4">Next Gen Fashion Intelligence Engine</p>
            <div className="flex items-center justify-center gap-8 text-sm text-cyan-300">
              <span className="flex items-center gap-2 bg-slate-800/50 px-3 py-1 rounded-full border border-cyan-500/30">
                <Eye className="h-4 w-4" />
                YOLOv8 Vision
              </span>
              <span className="flex items-center gap-2 bg-slate-800/50 px-3 py-1 rounded-full border border-purple-500/30">
                <Brain className="h-4 w-4" />
                CLIP Neural Net
              </span>
              <span className="flex items-center gap-2 bg-slate-800/50 px-3 py-1 rounded-full border border-pink-500/30">
                <Zap className="h-4 w-4" />
                FAISS Vector Search
              </span>
              <span className="flex items-center gap-2 bg-slate-800/50 px-3 py-1 rounded-full border border-cyan-500/30">
                <Database className="h-4 w-4" />
                200+ Products
              </span>
            </div>
          </div>
        </div>

        {/* Futuristic Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800/50 backdrop-blur-sm border border-cyan-500/20 rounded-xl p-1">
            <TabsTrigger
              value="single"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500 data-[state=active]:to-purple-600 data-[state=active]:text-white transition-all duration-300"
            >
              Video
            </TabsTrigger>
            <TabsTrigger
              value="batch"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-500 data-[state=active]:to-pink-600 data-[state=active]:text-white transition-all duration-300"
            >
              Batch Process
            </TabsTrigger>
            <TabsTrigger
              value="api"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-pink-500 data-[state=active]:to-cyan-600 data-[state=active]:text-white transition-all duration-300"
            >
              API Docs
            </TabsTrigger>
          </TabsList>

          {/* Single Video Processing */}
          <TabsContent value="single" className="space-y-6 mt-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Upload Section */}
              <Card className="bg-slate-800/30 backdrop-blur-sm border border-cyan-500/20 shadow-2xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-cyan-100">
                    <Upload className="h-5 w-5 text-cyan-400" />
                    Upload Video
                  </CardTitle>
                  <CardDescription className="text-cyan-300/70">
                    Upload 5-15s fashion video for AI analysis
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="border-2 border-dashed border-cyan-500/30 rounded-xl p-8 text-center bg-gradient-to-br from-slate-800/20 to-purple-900/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300">
                    <Input
                      type="file"
                      accept="video/*"
                      onChange={handleFileSelect}
                      className="hidden"
                      id="video-upload"
                    />
                    <label htmlFor="video-upload" className="cursor-pointer">
                      {selectedFile ? (
                        <div className="space-y-4 animate-fade-in">
                          <Play className="h-16 w-16 text-cyan-400 mx-auto animate-pulse" />
                          <div>
                            <p className="text-lg font-medium text-cyan-100">{selectedFile.name}</p>
                            <p className="text-sm text-cyan-300">
                              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for processing
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <Video className="h-16 w-16 text-cyan-400/60 mx-auto animate-bounce" />
                          <div>
                            <p className="text-lg font-medium text-cyan-100">Drop video file here</p>
                            <p className="text-sm text-cyan-300/70 mt-2">MP4, MOV, AVI • Max 50MB</p>
                          </div>
                        </div>
                      )}
                    </label>
                  </div>

                  <div className="space-y-3">
                    <label className="text-sm font-medium text-cyan-200">Caption Analysis</label>
                    <Textarea
                      placeholder="Add caption or hashtags for enhanced vibe detection..."
                      value={caption}
                      onChange={(e) => setCaption(e.target.value)}
                      rows={3}
                      className="bg-slate-800/50 border-cyan-500/30 text-cyan-100 placeholder:text-cyan-400/50 focus:border-cyan-400 transition-colors"
                    />
                  </div>

                  {processing && (
                    <div className="space-y-4 animate-fade-in">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-cyan-300 font-medium animate-pulse">{currentStep}</span>
                        <span className="text-cyan-400">{Math.round(progress)}%</span>
                      </div>
                      <div className="relative">
                        <Progress value={progress} className="h-3 bg-slate-700/50 border border-cyan-500/20" />
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-full animate-pulse"></div>
                      </div>
                    </div>
                  )}

                  <Button
                    onClick={processSingleVideo}
                    disabled={!selectedFile || processing}
                    className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 text-white font-bold py-4 rounded-xl shadow-lg hover:shadow-cyan-500/25 transition-all duration-300 transform hover:scale-105"
                    size="lg"
                  >
                    {processing ? (
                      <span className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        Processing...
                      </span>
                    ) : (
                      "Launch"
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Results Section */}
              {result && (
                <Card className="bg-slate-800/30 backdrop-blur-sm border border-purple-500/20 shadow-2xl animate-fade-in">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between text-purple-100">
                      <span className="flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-purple-400" />
                        Analysis Results
                      </span>
                      <Button
                        onClick={() => downloadResult(result, `${result.video_id}_result.json`)}
                        variant="outline"
                        size="sm"
                        className="border-purple-500/30 text-purple-300 hover:bg-purple-500/20"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Export JSON
                      </Button>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Vibes */}
                    <div className="space-y-3">
                      <h4 className="font-medium text-purple-200 flex items-center gap-2">
                        <Tag className="h-4 w-4" />
                        Detected Vibes
                      </h4>
                      <div className="flex flex-wrap gap-3">
                        {result.vibes.map((vibe, index) => (
                          <Badge
                            key={index}
                            className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-200 border border-purple-400/30 px-4 py-2 rounded-full backdrop-blur-sm hover:scale-105 transition-transform duration-200"
                          >
                            {vibe}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Products */}
                    <div className="space-y-3">
                      <h4 className="font-medium text-purple-200 flex items-center gap-2">
                        <ImageIcon className="h-4 w-4" />
                        Product Matches
                      </h4>
                      <div className="space-y-3">
                        {result.products.map((product, index) => (
                          <div
                            key={index}
                            className="bg-gradient-to-r from-slate-800/40 to-purple-900/20 border border-purple-500/20 rounded-xl p-4 backdrop-blur-sm hover:border-purple-400/40 transition-all duration-300"
                          >
                            <div className="flex items-center justify-between mb-3">
                              <span className="font-medium capitalize text-purple-100">{product.type}</span>
                              <Badge
                                className={
                                  product.match_type === "exact"
                                    ? "bg-green-500/20 text-green-300 border-green-400/30"
                                    : "bg-yellow-500/20 text-yellow-300 border-yellow-400/30"
                                }
                              >
                                {product.match_type === "exact" ? "Exact Match" : "Similar"}
                              </Badge>
                            </div>
                            <div className="grid grid-cols-2 gap-4 text-sm text-purple-300">
                              <div>
                                <p>
                                  Color: <span className="text-purple-100">{product.color}</span>
                                </p>
                                <p>
                                  ID: <span className="text-purple-100">{product.matched_product_id}</span>
                                </p>
                              </div>
                              <div>
                                <p>
                                  Confidence:{" "}
                                  <span className="text-purple-100">{(product.confidence * 100).toFixed(1)}%</span>
                                </p>
                                <div className="w-full bg-slate-700/50 rounded-full h-2 mt-1">
                                  <div
                                    className="bg-gradient-to-r from-cyan-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                                    style={{ width: `${product.confidence * 100}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* JSON Output */}
                    <div className="space-y-3">
                      <h4 className="font-medium text-purple-200">Raw Output</h4>
                      <pre className="bg-slate-900/80 border border-cyan-500/20 text-cyan-300 p-4 rounded-xl text-sm overflow-x-auto backdrop-blur-sm">
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Batch Processing */}
          <TabsContent value="batch" className="space-y-6 mt-8">
            <Card className="bg-slate-800/30 backdrop-blur-sm border border-purple-500/20 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-purple-100">Batch Processing</CardTitle>
                <CardDescription className="text-purple-300/70">
                  Process all 10 sample videos with parallel AI analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {processing && (
                  <div className="space-y-4 animate-fade-in">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-purple-300 font-medium animate-pulse">{currentStep}</span>
                      <span className="text-purple-400">{Math.round(progress)}%</span>
                    </div>
                    <div className="relative">
                      <Progress value={progress} className="h-3 bg-slate-700/50 border border-purple-500/20" />
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-full animate-pulse"></div>
                    </div>
                  </div>
                )}

                <Button
                  onClick={processBatchVideos}
                  disabled={processing}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white font-bold py-4 rounded-xl shadow-lg hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-105"
                  size="lg"
                >
                  {processing ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      Processing Batch...
                    </span>
                  ) : (
                    "Launch"
                  )}
                </Button>

                {batchResults && (
                  <div className="space-y-6 animate-fade-in">
                    {/* Stats Grid */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gradient-to-br from-green-500/20 to-emerald-600/20 border border-green-400/30 p-6 rounded-xl text-center backdrop-blur-sm">
                        <p className="text-3xl font-bold text-green-300">{batchResults.processed_videos}</p>
                        <p className="text-sm text-green-400">Videos Processed</p>
                      </div>
                      <div className="bg-gradient-to-br from-blue-500/20 to-cyan-600/20 border border-blue-400/30 p-6 rounded-xl text-center backdrop-blur-sm">
                        <p className="text-3xl font-bold text-blue-300">
                          {batchResults.results.reduce((acc, r) => acc + r.products.length, 0)}
                        </p>
                        <p className="text-sm text-blue-400">Products Detected</p>
                      </div>
                      <div className="bg-gradient-to-br from-purple-500/20 to-pink-600/20 border border-purple-400/30 p-6 rounded-xl text-center backdrop-blur-sm">
                        <p className="text-3xl font-bold text-purple-300">{batchResults.processing_time}s</p>
                        <p className="text-sm text-purple-400">Total Time</p>
                      </div>
                    </div>

                    <Button
                      onClick={downloadAllResults}
                      className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 transition-all duration-300"
                      variant="outline"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download All Results
                    </Button>

                    {/* Results Preview */}
                    <div className="space-y-3">
                      <h4 className="font-medium text-purple-200">Batch Results</h4>
                      <div className="max-h-96 overflow-y-auto space-y-2 pr-2">
                        {batchResults.results.map((result, index) => (
                          <div
                            key={index}
                            className="bg-slate-800/40 border border-cyan-500/20 rounded-lg p-3 backdrop-blur-sm hover:border-cyan-400/40 transition-all duration-200"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium text-cyan-100">{result.video_id}</span>
                              <div className="flex gap-2">
                                {result.vibes.map((vibe, i) => (
                                  <Badge
                                    key={i}
                                    variant="secondary"
                                    className="text-xs bg-purple-500/20 text-purple-300 border-purple-400/30"
                                  >
                                    {vibe}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            <p className="text-sm text-cyan-300/70 mt-1">{result.products.length} products detected</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* API Documentation */}
          <TabsContent value="api" className="space-y-6 mt-8">
            <Card className="bg-slate-800/30 backdrop-blur-sm border border-cyan-500/20 shadow-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-cyan-100">
                  <FileText className="h-5 w-5 text-cyan-400" />
                  API Interface
                </CardTitle>
                <CardDescription className="text-cyan-300/70">
                  FastAPI endpoints for production deployment
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-green-500/10 to-emerald-600/10 border border-green-400/30 rounded-xl p-4 backdrop-blur-sm">
                    <h4 className="font-medium text-green-300 mb-2">POST /process-video</h4>
                    <p className="text-sm text-green-400/80 mb-3">Video analysis endpoint</p>
                    <pre className="bg-slate-900/80 text-green-300 p-3 rounded-lg text-sm overflow-x-auto">
                      {`curl -X POST "http://localhost:8000/process-video" \\
  -F "video=@video.mp4" \\
  -F "caption=Futuristic Y2K metallic outfit "`}
                    </pre>
                  </div>

                  <div className="bg-gradient-to-r from-blue-500/10 to-cyan-600/10 border border-blue-400/30 rounded-xl p-4 backdrop-blur-sm">
                    <h4 className="font-medium text-blue-300 mb-2">GET /health</h4>
                    <p className="text-sm text-blue-400/80 mb-3">System health monitoring</p>
                    <pre className="bg-slate-900/80 text-cyan-300 p-3 rounded-lg text-sm overflow-x-auto">
                      {`curl -X GET "http://localhost:8000/health"`}
                    </pre>
                  </div>

                  <div className="bg-gradient-to-r from-purple-500/10 to-pink-600/10 border border-purple-400/30 rounded-xl p-4 backdrop-blur-sm">
                    <h4 className="font-medium text-purple-300 mb-2">GET /supported-vibes</h4>
                    <p className="text-sm text-purple-400/80 mb-3">Available vibe classifications</p>
                    <pre className="bg-slate-900/80 text-purple-300 p-3 rounded-lg text-sm overflow-x-auto">
                      {`curl -X GET "http://localhost:8000/supported-vibes"`}
                    </pre>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-cyan-500/10 to-blue-600/10 border border-cyan-400/30 rounded-xl p-4 backdrop-blur-sm">
                  <h4 className="font-medium text-cyan-300 mb-3">Response Format</h4>
                  <pre className="bg-slate-900/90 text-cyan-300 p-4 rounded-lg text-sm overflow-x-auto">
                    {`{
  "video_id": "abc123",
  "vibes": ["Y2K", "Futuristic"],
  "products": [
    {
      "type": "dress",
      "color": "metallic",
      "match_type": "exact",
      "matched_product_id": "prod_456",
      "confidence": 0.94
    }
  ]
}`}
                  </pre>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </div>
  )
}
