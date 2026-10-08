import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MinimalLayout } from "./App";
import "./index.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error('Không tìm thấy phần tử "#root" trong index.html.');
}

createRoot(rootElement).render(
  <StrictMode>
    <MinimalLayout />
  </StrictMode>,
);
