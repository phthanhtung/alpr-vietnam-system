# Cấu trúc Frontend

Frontend sử dụng React + TypeScript + Vite, thiết kế cho một trang nhận diện biển số duy nhất và gọi REST API của FastAPI.

```text
frontend/
├── .env.example
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
└── src/
    ├── App.tsx
    ├── main.tsx
    ├── index.css
    ├── hooks/
    │   └── useDetection.ts
    ├── services/
    │   └── detectionApi.ts
    ├── types/
    │   └── detection.ts
```

Không sử dụng router hoặc cấu trúc nhiều trang. Cấu hình URL backend qua biến `VITE_API_BASE_URL`.
