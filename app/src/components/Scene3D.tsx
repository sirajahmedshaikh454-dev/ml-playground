"use client";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function Cloud({ pts, colors, spin }: { pts: number[][]; colors: string[]; spin: boolean }) {
  const ref = useRef<THREE.Points>(null);
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    // normalize to a unit cube so any dataset frames nicely
    const lo = [0, 1, 2].map((j) => Math.min(...pts.map((p) => p[j] ?? 0)));
    const hi = [0, 1, 2].map((j) => Math.max(...pts.map((p) => p[j] ?? 0)));
    const pos = new Float32Array(pts.length * 3);
    const col = new Float32Array(pts.length * 3);
    const c = new THREE.Color();
    pts.forEach((p, i) => {
      for (let j = 0; j < 3; j++) pos[i * 3 + j] = (((p[j] ?? 0) - lo[j]) / (hi[j] - lo[j] || 1) - 0.5) * 4;
      c.set(colors[i] ?? "#60a5fa");
      col.set([c.r, c.g, c.b], i * 3);
    });
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return g;
  }, [pts, colors]);
  useFrame((_, dt) => { if (spin && ref.current) ref.current.rotation.y += dt * 0.15; });
  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial size={0.09} vertexColors sizeAttenuation transparent opacity={0.95} />
    </points>
  );
}

export default function PointCloud3D({ pts, colors, height = 340, label }: { pts: number[][]; colors: string[]; height?: number; label: string }) {
  const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return (
    <div style={{ height }} className="grid-bg overflow-hidden rounded-xl border border-border" role="img" aria-label={label}>
      <Canvas camera={{ position: [4.5, 3, 4.5], fov: 50 }} dpr={[1, 2]}>
        <Cloud pts={pts} colors={colors} spin={!reduce} />
        <gridHelper args={[4, 8, "#334155", "#1e293b"]} position={[0, -2, 0]} />
        <OrbitControls enablePan={false} makeDefault />
      </Canvas>
    </div>
  );
}
