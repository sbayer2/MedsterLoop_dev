"use client";

import { useState } from "react";
import { useSession, signOut } from "next-auth/react";
import { motion } from "framer-motion";
import {
    LogOut,
    MessageSquare,
    Activity,
    FileText,
    Settings,
    Menu,
    X
} from "lucide-react";
import ChatWindow from "@/components/ChatWindow";

export default function DashboardPage() {
    const { data: session } = useSession();
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [activeTab, setActiveTab] = useState("chat");

    return (
        <div className="flex h-screen bg-gray-900 text-white overflow-hidden">
            {/* Sidebar */}
            <motion.div
                animate={{ width: isSidebarOpen ? 256 : 80 }}
                className="bg-gray-800 border-r border-gray-700 flex flex-col transition-all duration-300 relative z-20"
            >
                <div className="p-4 flex items-center justify-between">
                    {isSidebarOpen && (
                        <span className="font-bold text-xl bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                            Medster
                        </span>
                    )}
                    <button
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                        className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                </div>

                <nav className="flex-1 px-2 py-4 space-y-2">
                    <SidebarItem
                        icon={<MessageSquare size={20} />}
                        label="Clinical Chat"
                        isOpen={isSidebarOpen}
                        active={activeTab === "chat"}
                        onClick={() => setActiveTab("chat")}
                    />
                    <SidebarItem
                        icon={<Activity size={20} />}
                        label="Patient Vitals"
                        isOpen={isSidebarOpen}
                        active={activeTab === "vitals"}
                        onClick={() => setActiveTab("vitals")}
                    />
                    <SidebarItem
                        icon={<FileText size={20} />}
                        label="Reports"
                        isOpen={isSidebarOpen}
                        active={activeTab === "reports"}
                        onClick={() => setActiveTab("reports")}
                    />
                    <SidebarItem
                        icon={<Settings size={20} />}
                        label="Settings"
                        isOpen={isSidebarOpen}
                        active={activeTab === "settings"}
                        onClick={() => setActiveTab("settings")}
                    />
                </nav>

                <div className="p-4 border-t border-gray-700">
                    <button
                        onClick={() => signOut({ callbackUrl: "/login" })}
                        className="flex items-center w-full p-2 text-red-400 hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <LogOut size={20} />
                        {isSidebarOpen && <span className="ml-3">Sign Out</span>}
                    </button>
                </div>
            </motion.div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-full overflow-hidden">
                <header className="bg-gray-800 border-b border-gray-700 p-4 flex justify-between items-center">
                    <h2 className="text-lg font-semibold">
                        {activeTab === "chat" && "Clinical Assistant"}
                        {activeTab === "vitals" && "Patient Vitals Dashboard"}
                        {activeTab === "reports" && "Medical Reports"}
                        {activeTab === "settings" && "System Settings"}
                    </h2>
                    <div className="flex items-center space-x-4">
                        <div className="text-sm text-gray-400">
                            Logged in as <span className="text-white font-medium">{session?.user?.name || "User"}</span>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center font-bold">
                            {session?.user?.name?.[0] || "U"}
                        </div>
                    </div>
                </header>

                <main className="flex-1 overflow-hidden relative">
                    {activeTab === "chat" && <ChatWindow />}
                    {activeTab !== "chat" && (
                        <div className="flex items-center justify-center h-full text-gray-500">
                            Feature coming soon...
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}

function SidebarItem({ icon, label, isOpen, active, onClick }: any) {
    return (
        <button
            onClick={onClick}
            className={`w-full flex items-center p-3 rounded-lg transition-all ${active
                    ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
                    : "text-gray-400 hover:bg-gray-700 hover:text-white"
                }`}
        >
            {icon}
            {isOpen && <span className="ml-3 font-medium">{label}</span>}
        </button>
    );
}
