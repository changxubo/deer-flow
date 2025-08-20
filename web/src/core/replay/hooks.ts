// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { useSearchParams } from "next/navigation";
import { useMemo } from "react";

import { env } from "~/env";

import { extractFromSearchParams } from "./get-replay-id";

export function useReplay() {
  const searchParams = useSearchParams();
  const replayId = useMemo(
    () => extractFromSearchParams(searchParams.toString(), "replay"),
    [searchParams],
  );
  return {
    isReplay: replayId != null || env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY,
    replayId,
  };
}
