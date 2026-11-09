import { buildDataset, } from "./ingest";
import { type Dataset, gauss, mulberry32 } from "./core";

type R = Record<string, unknown>;

// Deliberately messy: missing cells, duplicates, outliers -> shows the cleaning step off.
export function irisMessy(): Dataset {
  const r = mulberry32(7);
  const sp: [string, number[]][] = [
    ["setosa", [5.0, 3.4, 1.5, 0.25]],
    ["versicolor", [5.9, 2.8, 4.3, 1.3]],
    ["virginica", [6.6, 3.0, 5.5, 2.0]],
  ];
  const rows: R[] = [];
  for (const [name, m] of sp)
    for (let i = 0; i < 60; i++)
      rows.push({
        sepal_length: +(m[0] + gauss(r) * 0.4).toFixed(2),
        sepal_width: +(m[1] + gauss(r) * 0.3).toFixed(2),
        petal_length: +(m[2] + gauss(r) * 0.4).toFixed(2),
        petal_width: +(m[3] + gauss(r) * 0.15).toFixed(2),
        species: name,
      });
  const keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
  rows.forEach((row) => keys.forEach((k) => { if (r() < 0.04) row[k] = ""; }));
  for (let i = 0; i < 5; i++) rows.push({ ...rows[Math.floor(r() * 100)] });
  [10, 70, 150].forEach((i) => (rows[i].sepal_length = 45 + i / 10));
  return buildDataset("iris_messy.csv", [...keys, "species"], rows, "Sample: 3-class flowers with missing values, duplicates and outliers injected.");
}

export function moons(): Dataset {
  const r = mulberry32(11);
  const rows: R[] = [];
  for (let i = 0; i < 400; i++) {
    const c = i % 2;
    const t = Math.PI * r();
    const x = c ? 1 - Math.cos(t) : Math.cos(t);
    const y = c ? 0.5 - Math.sin(t) : Math.sin(t);
    rows.push({ x: +(x + gauss(r) * 0.15).toFixed(3), y: +(y + gauss(r) * 0.15).toFixed(3), class: c ? "B" : "A" });
  }
  return buildDataset("moons.csv", ["x", "y", "class"], rows, "Sample: two interleaved moons — linear models fail, non-linear ones win.");
}

export function houses(): Dataset {
  const r = mulberry32(21);
  const hoods = ["Downtown", "Suburb", "Lakeside", "Industrial"];
  const bump = [60, 10, 45, -25];
  const rows: R[] = [];
  for (let i = 0; i < 500; i++) {
    const area = 40 + r() * 200;
    const beds = Math.max(1, Math.round(area / 55 + gauss(r) * 0.6));
    const age = Math.floor(r() * 60);
    const dist = +(r() * 25).toFixed(1);
    const h = Math.floor(r() * 4);
    const price = 40 + area * 1.9 + beds * 8 - age * 0.9 - dist * 2.5 + bump[h] + (area / 100) ** 2 * 12 + gauss(r) * 14;
    rows.push({ area_m2: +area.toFixed(1), bedrooms: beds, age_years: age, km_to_center: dist, neighborhood: hoods[h], price_k: +price.toFixed(1) });
  }
  rows.forEach((row) => { if (r() < 0.05) row.age_years = ""; if (r() < 0.03) row.neighborhood = ""; });
  [5, 200, 333].forEach((i) => (rows[i].price_k = 4000 + i));
  return buildDataset("houses.csv", ["area_m2", "bedrooms", "age_years", "km_to_center", "neighborhood", "price_k"], rows, "Sample: predict house price (regression) with missing values and outliers.");
}

export const SAMPLES = [
  { id: "iris", title: "Messy Iris", blurb: "3-class classification, missing values + outliers", make: irisMessy },
  { id: "moons", title: "Two Moons", blurb: "Non-linear 2D classification", make: moons },
  { id: "houses", title: "House Prices", blurb: "Regression with categorical features", make: houses },
];
