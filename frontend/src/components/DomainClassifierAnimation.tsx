"use client"

import { useEffect, useState, useRef } from "react"
import { Shield, ShieldAlert } from "lucide-react"
import { cn } from "@/lib/utils"

interface Domain {
  id: number
  name: string
  isClean: boolean
  x: number
  y: number
  targetX: number
  targetY: number
  state: "entering" | "classifying" | "moving" | "done"
}

const dgaDomains = [
  "qx7k2mnp9v.com",
  "z9ht4wbry.net",
  "p3nfg8xqj.org",
  "v5kl2yhxt.com",
  "m8wq3zpfn.net",
  "j4rx7ngkt.org",
  "c2vl9bphx.com",
  "n6zy4kwqf.net",
  "t1sd8jvrc.org",
  "h7qk2mnp9v.com",
  "y3ht4wbry.net",
]

const cleanDomains = [
  "google.com",
  "github.com",
  "vercel.com",
  "stackoverflow.com",
  "mozilla.org",
  "wikipedia.org",
  "amazon.com",
  "microsoft.com",
  "apple.com",
  "netflix.com",
  "spotify.com",
]

export default function DomainClassifierAnimation() {
  const [domains, setDomains] = useState<Domain[]>([])
  const nextIdRef = useRef(0)

  useEffect(() => {
    const addDomain = () => {
      const isClean = Math.random() > 0.5
      const domainList = isClean ? cleanDomains : dgaDomains
      const domainName = domainList[Math.floor(Math.random() * domainList.length)]

      const newDomain: Domain = {
        id: nextIdRef.current,
        name: domainName,
        isClean,
        x: 50,
        y: 10,
        targetX: isClean ? 75 : 25,
        targetY: 85,
        state: "entering",
      }

      nextIdRef.current += 1
      setDomains((prev) => [...prev, newDomain])

      setTimeout(() => {
        setDomains((prev) => prev.map((d) => (d.id === newDomain.id ? { ...d, state: "classifying" } : d)))
      }, 800)

      setTimeout(() => {
        setDomains((prev) => prev.map((d) => (d.id === newDomain.id ? { ...d, state: "moving" } : d)))
      }, 2500)

      setTimeout(() => {
        setDomains((prev) => prev.map((d) => (d.id === newDomain.id ? { ...d, state: "done" } : d)))
      }, 3000)

      setTimeout(() => {
        setDomains((prev) => prev.filter((d) => d.id !== newDomain.id))
      }, 4000)
    }

    const initialTimeout = setTimeout(addDomain, 500)

    const interval = setInterval(addDomain, 2500)

    return () => {
      clearTimeout(initialTimeout)
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="relative w-full h-full bg-card/50 rounded-2xl border border-border/50 overflow-hidden backdrop-blur-sm">
      {/* Analysis zone */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-primary/10 border border-primary/30 rounded-lg px-4 py-2 text-sm font-mono text-primary">
        Analyzing...
      </div>

      {/* Safe zone */}
      <div className="absolute bottom-4 right-4 w-32 h-32 bg-accent/10 border-2 border-primary rounded-xl flex flex-col items-center justify-center gap-2">
        <Shield className="w-8 h-8 text-primary" />
        <span className="text-xs font-semibold text-primary">Safe</span>
      </div>

      {/* Malicious zone */}
      <div className="absolute bottom-4 left-4 w-32 h-32 bg-destructive/10 border-2 border-destructive rounded-xl flex flex-col items-center justify-center gap-2">
        <ShieldAlert className="w-8 h-8 text-destructive" />
        <span className="text-xs font-semibold text-destructive">Malicious</span>
      </div>

      {/* Animated domains */}
      {domains.map((domain) => (
        <div
          key={domain.id}
          className={cn(
            "absolute transition-all duration-1000 ease-out",
            domain.state === "entering" && "opacity-0 scale-50",
            domain.state === "classifying" && "opacity-100 scale-100",
            domain.state === "moving" && "opacity-100",
            domain.state === "done" && "opacity-0",
          )}
          style={{
            left: `${domain.state === "entering" || domain.state === "classifying" ? domain.x : domain.targetX}%`,
            top: `${domain.state === "entering" || domain.state === "classifying" ? domain.y : domain.targetY}%`,
            transform: "translate(-50%, -50%)",
          }}
        >
          <div
            className={cn(
              "px-3 py-2 rounded-lg border-2 font-mono text-xs whitespace-nowrap shadow-lg transition-colors duration-500",
              domain.state === "classifying" && "bg-card border-primary text-foreground animate-pulse",
              domain.state === "moving" && domain.isClean && "bg-primary/20 border-primary text-primary",
              domain.state === "moving" && !domain.isClean && "bg-destructive/20 border-destructive text-destructive",
            )}
          >
            {domain.name}
          </div>
        </div>
      ))}

      {/* Background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#8882_1px,transparent_1px),linear-gradient(to_bottom,#8882_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20" />
    </div>
  )
}
