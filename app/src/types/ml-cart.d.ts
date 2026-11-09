declare module "ml-cart" {
  interface TreeOptions {
    gainFunction?: "gini" | "regression";
    maxDepth?: number;
    minNumSamples?: number;
  }
  export class DecisionTreeClassifier {
    constructor(options?: TreeOptions);
    train(X: number[][], y: number[]): void;
    predict(X: number[][]): number[];
  }
  export class DecisionTreeRegression {
    constructor(options?: TreeOptions);
    train(X: number[][], y: number[]): void;
    predict(X: number[][]): number[];
  }
}
