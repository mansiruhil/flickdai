import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'flickdai',
  description: 'Next Gen Fashion Intelligence Engine',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
