import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const handler = NextAuth({
    providers: [
        CredentialsProvider({
            name: "Credentials",
            credentials: {
                username: { label: "Username", type: "text", placeholder: "admin" },
                password: { label: "Password", type: "password" }
            },
            async authorize(credentials, req) {
                // Add logic here to look up the user from the credentials supplied
                // For this demo, we'll just check hardcoded values
                const user = { id: "1", name: "Dr. Medster", email: "admin@medster.ai" };

                if (credentials?.username === "admin" && credentials?.password === "admin") {
                    return user;
                } else {
                    return null;
                }
            }
        })
    ],
    pages: {
        signIn: '/login',
    },
    callbacks: {
        async jwt({ token, user }) {
            return token;
        },
        async session({ session, token }) {
            return session;
        }
    }
});

export { handler as GET, handler as POST };
