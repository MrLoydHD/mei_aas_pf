"use client"

import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import { GoogleLogin, type CredentialResponse } from "@react-oauth/google"
import { Shield, Zap, Lock, ArrowRight, ShieldCheck, Database, Globe } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { useAuth } from "@/contexts/AuthContext"
import DomainClassifierAnimation from "@/components/DomainClassifierAnimation.tsx"
import { Cover } from "@/components/ui/cover";

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
}

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
}

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 }
}

export default function Home() {
  const navigate = useNavigate()
  const { user, login, isLoading } = useAuth()

  const handleGoogleSuccess = async (credentialResponse: CredentialResponse) => {
    if (credentialResponse.credential) {
      try {
        await login(credentialResponse.credential)
        navigate("/dashboard")
      } catch (error) {
        console.error("Login failed:", error)
      }
    }
  }

  const features = [
    {
      icon: Zap,
      title: "Real-time Analysis",
      description:
        "Instant domain classification using CNN-LSTM models with 95%+ accuracy for immediate threat detection.",
    },
    {
      icon: Shield,
      title: "Advanced ML Models",
      description:
        "Powered by deep learning to identify patterns in algorithmically generated domains used by botnets.",
    },
    {
      icon: Lock,
      title: "Sync Across Devices",
      description: "Seamlessly sync detection history between your dashboard and browser extension with cloud backup.",
    },
    {
      icon: Database,
      title: "Historical Analytics",
      description: "Track trends and patterns over time with comprehensive analytics and detailed reporting dashboard.",
    },
    {
      icon: Globe,
      title: "Browser Extension",
      description:
        "Protect your browsing in real-time with our lightweight extension that works across all major browsers.",
    },
  ]

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <motion.div
      className="min-h-screen"
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-12 lg:py-20">
        <motion.div
          className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center min-h-[80vh]"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          {/* Left: Content */}
          <motion.div className="space-y-8" variants={fadeInUp} transition={{ duration: 0.5 }}>
            <motion.div
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>AI-Powered Threat Detection</span>
            </motion.div>

            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight text-balance">
                Detect DGA
                <br />
                Domains in
                <br />
                <Cover className="text-primary">Real-Time</Cover>
              </h1>

              <p className="text-xl text-muted-foreground leading-relaxed max-w-xl text-pretty">
                Machine learning-powered detection of algorithmically generated domains. Protect your network from
                botnet C2 communication and malware infrastructure with 95%+ accuracy.
              </p>
            </motion.div>

            {/* CTA Section */}
            <motion.div
              className="space-y-6 pt-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {user ? (
                <div className="flex flex-col sm:flex-row gap-4 items-start">
                  <Button size="lg" className="text-base h-12 px-8" onClick={() => navigate("/dashboard")}>
                    Open Dashboard
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                  <div className="flex items-center gap-3 px-4">
                    {user.picture ? (
                      <img
                        src={user.picture || "/placeholder.svg"}
                        alt={user.name || "User"}
                        className="w-10 h-10 rounded-full border-2 border-border"
                      />
                    ) : (
                      <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center border-2 border-border">
                        <span className="text-primary-foreground font-semibold text-sm">
                          {(user.name || user.email)[0].toUpperCase()}
                        </span>
                      </div>
                    )}
                    <div className="text-sm">
                      <p className="font-medium text-foreground">{user.name || "User"}</p>
                      <p className="text-muted-foreground">{user.email}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col sm:flex-row gap-4">
                  <div className="flex justify-start">
                    <GoogleLogin
                      onSuccess={handleGoogleSuccess}
                      onError={() => console.error("Google login failed")}
                      size="large"
                      text="signin_with"
                    />
                  </div>
                  <Button
                    variant="outline"
                    size="lg"
                    className="text-base h-12 px-8 bg-transparent"
                    onClick={() => navigate("/dashboard")}
                  >
                    Try as Guest
                  </Button>
                </div>
              )}

              <p className="text-sm text-muted-foreground">Sign in to sync your detection history across devices</p>
            </motion.div>
          </motion.div>

          {/* Right: Animation */}
          <motion.div
            className="relative h-[500px] lg:h-[600px]"
            initial={{ opacity: 0, scale: 0.95, x: 30 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
          >
            <DomainClassifierAnimation />
          </motion.div>
        </motion.div>

        <motion.div
          className="mt-24 max-w-6xl mx-auto"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="relative overflow-hidden">
            {/* Fade out gradient on left edge */}
            <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none" />

            {/* Fade out gradient on right edge */}
            <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none" />

            {/* Infinite scrolling container */}
            <div className="flex gap-6 animate-infiniteScroll">
              {/* First set of cards */}
              {features.map((feature, index) => {
                const Icon = feature.icon
                return (
                  <Card
                    key={`first-${index}`}
                    className="border-border/50 bg-card/50 backdrop-blur-sm min-w-[300px]"
                  >
                    <CardContent className="pt-8 pb-8 text-center">
                      <div className="flex justify-center mb-4">
                        <div className="p-3 bg-primary/10 rounded-xl">
                          <Icon className="h-10 w-10 text-primary" />
                        </div>
                      </div>
                      <h3 className="font-semibold text-xl mb-3 text-foreground">{feature.title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
                    </CardContent>
                  </Card>
                )
              })}
              {/* Duplicate set for seamless loop */}
              {features.map((feature, index) => {
                const Icon = feature.icon
                return (
                  <Card
                    key={`second-${index}`}
                    className="border-border/50 bg-card/50 backdrop-blur-sm min-w-[350px]"
                  >
                    <CardContent className="pt-8 pb-8 text-center">
                      <div className="flex justify-center mb-4">
                        <div className="p-3 bg-primary/10 rounded-xl">
                          <Icon className="h-10 w-10 text-primary" />
                        </div>
                      </div>
                      <h3 className="font-semibold text-xl mb-3 text-foreground">{feature.title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}
