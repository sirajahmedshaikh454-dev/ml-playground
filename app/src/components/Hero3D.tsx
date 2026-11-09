"use client";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

// A "neural" sphere: Fibonacci-distributed nodes, nearest neighbours wired up.
function Net() {
  const group = useRef<THREE.Group>(null);
  const { pos, col, lines } = useMemo(() => {
    const n = 220, pts: THREE.Vector3[] = [];
    for (let i = 0; i < n; i++) {
      const y = 1 - (i / (n - 1)) * 2, r = Math.sqrt(1 - y * y), t = i * Math.PI * (3 - Math.sqrt(5));
      pts.push(new THREE.Vector3(Math.cos(t) * r, y, Math.sin(t) * r).multiplyScalar(2.1));
    }
    const pos = new Float32Array(pts.flatMap((p) => [p.x, p.y, p.z]));
    const a = new THREE.Color("#3b82f6"), b = new THREE.Color("#f59e0b");
    const col = new Float32Array(pts.flatMap((p) => { const c = a.clone().lerp(b, (p.y + 2.1) / 4.2); return [c.r, c.g, c.b]; }));
    const seg: number[] = [];
    pts.forEach((p, i) => pts.forEach((q, j) => { if (j > i && p.distanceTo(q) < 0.55) seg.push(p.x, p.y, p.z, q.x, q.y, q.z); }));
    return { pos, col, lines: new Float32Array(seg) };
  }, []);
  useFrame((s, dt) => {
    if (!group.current) return;
    group.current.rotation.y += dt * 0.12;
    group.current.rotation.x = Math.sin(s.clock.elapsedTime * 0.2) * 0.15;
  });
  return (
    <group ref={group}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pos, 3]} />
          <bufferAttribute attach="attributes-color" args={[col, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.07} vertexColors sizeAttenuation />
      </points>
      <lineSegments>
        <bufferGeometry><bufferAttribute attach="attributes-position" args={[lines, 3]} /></bufferGeometry>
        <lineBasicMaterial color="#60a5fa" transparent opacity={0.18} />
      </lineSegments>
    </group>
  );
}

export default function Hero3D() {
  const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return (
    <div className="absolute inset-0" aria-hidden>
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }} dpr={[1, 2]} frameloop={reduce ? "demand" : "always"}>
        <Net />
      </Canvas>
    </div>
  );
}
