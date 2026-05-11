declare module "node:sqlite" {
  export class DatabaseSync {
    constructor(path?: string, options?: { open?: boolean });

    close(): void;
    prepare(sql: string): {
      all: (...parameters: unknown[]) => unknown[];
    };
  }
}
