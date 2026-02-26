# Étape 1 : Construction (Build)
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Étape 2 : Service (Production)
FROM nginx:stable-alpine
# Pour React (CRA), le dossier de sortie est 'build'. Pour Vite, c'est 'dist'.
# Vérifie ton dossier après un 'npm run build' local.
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]