"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, Bot, User, Loader2, X, FileText } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    timestamp: Date;
    fileName?: string;
}

export default function ChatWindow() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "1",
            role: "assistant",
            content: "Hello, Dr. Medster. I am your clinical assistant powered by Claude Sonnet 4.5. How can I help you analyze patient data today?\n\nYou can also upload clinical documents (PDF, TXT, CSV, JSON) using the paperclip icon for analysis.",
            timestamp: new Date(),
        }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [fileContent, setFileContent] = useState<string>("");
    const [fileTruncated, setFileTruncated] = useState<boolean>(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Max content size (~50K tokens = ~150K characters to stay under Claude's 200K limit)
    const MAX_CONTENT_CHARS = 150000;

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Check file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            alert("File size must be less than 5MB");
            return;
        }

        // Check file type
        const allowedTypes = ['.txt', '.pdf', '.csv', '.json', '.md', '.xml', '.hl7'];
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!allowedTypes.includes(fileExt)) {
            alert(`Supported file types: ${allowedTypes.join(', ')}`);
            return;
        }

        setSelectedFile(file);
        setFileTruncated(false);

        // Read file content
        const reader = new FileReader();
        reader.onload = (event) => {
            let content = event.target?.result as string;

            // Truncate if too large for Claude's context window
            if (content.length > MAX_CONTENT_CHARS) {
                content = content.substring(0, MAX_CONTENT_CHARS) +
                    "\n\n[... FILE TRUNCATED - Showing first " +
                    Math.round(MAX_CONTENT_CHARS / 1000) + "K characters of " +
                    Math.round(file.size / 1000) + "K total ...]";
                setFileTruncated(true);
            }
            setFileContent(content);
        };
        reader.readAsText(file);
    };

    const clearFile = () => {
        setSelectedFile(null);
        setFileContent("");
        setFileTruncated(false);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if ((!input.trim() && !selectedFile) || isLoading) return;

        // Build message content
        let messageContent = input.trim();
        let displayContent = input.trim();

        if (selectedFile && fileContent) {
            const filePrompt = messageContent
                ? `${messageContent}\n\n--- Attached File: ${selectedFile.name} ---\n${fileContent}`
                : `Please analyze the following clinical document:\n\n--- File: ${selectedFile.name} ---\n${fileContent}`;
            messageContent = filePrompt;
            displayContent = messageContent
                ? `${input.trim()}\n\n📎 Attached: ${selectedFile.name}`
                : `📎 Uploaded: ${selectedFile.name} for analysis`;
        }

        const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: displayContent,
            timestamp: new Date(),
            fileName: selectedFile?.name,
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        clearFile();
        setIsLoading(true);

        try {
            const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
            const response = await fetch(`${backendUrl}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageContent }),
            });

            if (!response.ok) throw new Error("Failed to get response");

            const data = await response.json();

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: data.response,
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, botMessage]);
        } catch (error) {
            console.error("Error:", error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: "I apologize, but I encountered an error connecting to the clinical backend. Please ensure the backend server is running.",
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-gray-900">
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
                <AnimatePresence>
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            <div
                                className={`max-w-[80%] rounded-2xl p-4 shadow-md ${msg.role === "user"
                                    ? "bg-blue-600 text-white rounded-br-none"
                                    : "bg-gray-800 text-gray-100 rounded-bl-none border border-gray-700"
                                    }`}
                            >
                                <div className="flex items-center gap-2 mb-1 opacity-70 text-xs">
                                    {msg.role === "user" ? <User size={12} /> : <Bot size={12} />}
                                    <span>{msg.role === "user" ? "You" : "Medster AI"}</span>
                                    <span>•</span>
                                    <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                </div>
                                <div className="whitespace-pre-wrap leading-relaxed">
                                    {msg.content}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex justify-start"
                    >
                        <div className="bg-gray-800 rounded-2xl p-4 rounded-bl-none border border-gray-700 flex items-center gap-3">
                            <Loader2 className="animate-spin text-blue-400" size={20} />
                            <span className="text-gray-400 text-sm">Analyzing clinical data...</span>
                        </div>
                    </motion.div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="p-4 bg-gray-800 border-t border-gray-700">
                {/* File preview */}
                {selectedFile && (
                    <div className="max-w-4xl mx-auto mb-2">
                        <div className={`inline-flex items-center gap-2 ${fileTruncated ? 'bg-yellow-900/50 border border-yellow-600' : 'bg-gray-700'} text-gray-200 px-3 py-2 rounded-lg text-sm`}>
                            <FileText size={16} className={fileTruncated ? "text-yellow-400" : "text-blue-400"} />
                            <span className="truncate max-w-[200px]">{selectedFile.name}</span>
                            <span className="text-gray-400">({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                            {fileTruncated && (
                                <span className="text-yellow-400 text-xs">(truncated to 150K chars)</span>
                            )}
                            <button
                                type="button"
                                onClick={clearFile}
                                className="ml-1 p-1 hover:bg-gray-600 rounded transition-colors"
                                title="Remove file"
                            >
                                <X size={14} />
                            </button>
                        </div>
                    </div>
                )}
                <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto">
                    {/* Hidden file input */}
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileSelect}
                        accept=".txt,.pdf,.csv,.json,.md,.xml,.hl7"
                        className="hidden"
                    />
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={selectedFile ? "Add a message or press send to analyze..." : "Ask a clinical question..."}
                        className="w-full bg-gray-900 border border-gray-600 rounded-xl py-4 pl-4 pr-24 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all shadow-inner"
                    />
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className={`p-2 rounded-lg transition-colors ${selectedFile ? 'text-blue-400 bg-gray-700' : 'text-gray-400 hover:text-white hover:bg-gray-700'}`}
                            title="Attach file (TXT, PDF, CSV, JSON, MD, XML, HL7)"
                        >
                            <Paperclip size={20} />
                        </button>
                        <button
                            type="submit"
                            disabled={(!input.trim() && !selectedFile) || isLoading}
                            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg"
                        >
                            <Send size={20} />
                        </button>
                    </div>
                </form>
                <p className="text-center text-xs text-gray-500 mt-2">
                    Medster AI can make mistakes. Verify important clinical information.
                </p>
            </div>
        </div>
    );
}
