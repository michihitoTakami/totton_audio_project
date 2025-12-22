(() => {
  const CONTACT_URL =
    "https://docs.google.com/forms/d/e/1FAIpQLSc164tZdiH9-2xHKnkegs8amf0CdZmB1vjiSA34ckSL6ZiGmw/viewform";

  const DICT = {
    en: {
      nav: { features: "Features", arch: "Architecture", contact: "Contact" },
      cta: {
        primary: "Contact",
        explore: "Explore features",
        contact: "Get in touch",
      },
      hero: {
        eyebrow: "Ultra high-resolution realtime audio processing",
        titleA: "A small box,",
        titleB: "a huge leap in sound.",
        lead:
          "Totton Audio delivers a GPU-accelerated realtime audio pipeline for headphone users—ultra high-resolution upsampling, headphone EQ correction, and a simple control UI.",
      },
      meta: {
        gpu: "GPU FFT convolution (CUDA)",
        minphase: "Minimum phase FIR (640k taps)",
        ui: "FastAPI control plane",
      },
      features: {
        title: "What this project does",
        subtitle:
          "Designed for ultimate simplicity: connect the box, click a few settings, and enjoy better sound.",
        f1: {
          title: "GPU-accelerated upsampling",
          text:
            "Realtime upsampling powered by GPU convolution and high-quality resampling—built for low latency and stability.",
        },
        f2: {
          title: "640k-tap minimum-phase FIR",
          text:
            "A long-tap minimum-phase FIR filter to achieve strong stopband attenuation while avoiding pre-ringing.",
        },
        f3: {
          title: "Headphone EQ correction",
          text:
            "Headphone correction workflow leveraging OPRA data (CC BY-SA 4.0) and a target curve—aimed at consistent, pleasant tonality.",
        },
        f4: {
          title: "Standalone DDC/DSP device",
          text:
            "Runs on Jetson Orin Nano (production) or PC (development) as a self-contained audio processing device.",
        },
        f5: {
          title: "Simple control UI",
          text:
            "Control plane built with Python/FastAPI, designed for a clean web UI and operational clarity.",
        },
        f6: {
          title: "Engine & streaming building blocks",
          text:
            "Data plane in C++/CUDA, plus interfaces like ZeroMQ and RTP notes for practical deployment and integration.",
        },
      },
      arch: {
        title: "Architecture",
        subtitle: "Split into a friendly control plane and a fast data plane.",
        diagramTitle: "High-level layout",
        diagram:
          "Control Plane (Python/FastAPI)     Data Plane (C++ Audio Engine)\n" +
          "├── IR Generator (scipy)           ├── GPU FFT Convolution (CUDA)\n" +
          "├── OPRA Integration               ├── libsoxr Resampling\n" +
          "└── ZeroMQ Command Interface   <-> └── ALSA Output\n",
        roadmapTitle: "Roadmap (high-level)",
        steps: {
          s1: { k: "Phase 1", v: "Core engine & middleware (in progress)" },
          s2: { k: "Phase 2", v: "Control plane & Web UI" },
          s3: { k: "Phase 3", v: "Jetson hardware integration" },
        },
        note:
          "Totton Audio is being developed as an independent project, but collaboration for productization, hardware development, and DSP/AI support is possible.",
      },
      contact: {
        title: "Contact",
        subtitle:
          "Partnerships, licensing, PoC/demo listening, or technical consultation—feel free to reach out.",
        cardTitle: "Let’s build the next-level listening experience",
        cardText:
          "Use the contact form below. We’ll handle your inquiry carefully and only use it for communication regarding this project.",
        form: "Open contact form",
        copy: "Copy link",
        hint:
          "Tip: If your browser blocks pop-ups, right-click the button and open in a new tab.",
        copied: "Copied the contact form link.",
        copyFailed: "Could not copy. Please copy the link manually.",
      },
      footer: {
        copy:
          "© Totton Audio. Brand page draft for this repository.\nLanguage defaults to English unless your browser is set to Japanese.",
        backToTop: "Back to top",
      },
    },
    ja: {
      nav: { features: "特徴", arch: "アーキテクチャ", contact: "お問い合わせ" },
      cta: {
        primary: "お問い合わせ",
        explore: "特徴を見る",
        contact: "相談する",
      },
      hero: {
        eyebrow: "超高解像度・リアルタイム音声処理",
        titleA: "小さな箱で、",
        titleB: "音を大きく変える。",
        lead:
          "Totton Audio は、ヘッドホンユーザー向けのGPU加速リアルタイム音声処理システムです。超高解像度アップサンプリング、ヘッドホンEQ補正、そしてシンプルな管理UIを目指します。",
      },
      meta: {
        gpu: "GPU FFT畳み込み（CUDA）",
        minphase: "最小位相FIR（640k taps）",
        ui: "FastAPIコントロール",
      },
      features: {
        title: "このリポジトリの主な機能",
        subtitle:
          "究極のシンプルさ：箱をつなぐ → 管理画面でポチポチ → 最高の音。",
        f1: {
          title: "GPU加速アップサンプリング",
          text:
            "GPU畳み込みと高品質リサンプルにより、低遅延・安定動作を意識したリアルタイム処理を実現します。",
        },
        f2: {
          title: "640k taps 最小位相FIR",
          text:
            "長タップの最小位相FIRで強い阻止帯域減衰を狙いつつ、プリリンギングを避ける設計です。",
        },
        f3: {
          title: "ヘッドホンEQ補正",
          text:
            "OPRAデータ（CC BY-SA 4.0）とターゲットカーブを使った補正フローにより、自然で一貫した音作りを目指します。",
        },
        f4: {
          title: "スタンドアロンDDC/DSP",
          text:
            "Jetson Orin Nano（本番）またはPC（開発）で動作する、自己完結したDDC/DSPデバイス構成です。",
        },
        f5: {
          title: "シンプルな管理UI",
          text:
            "Python/FastAPIのコントロールプレーンで、見通しの良いWeb UIと運用性を重視します。",
        },
        f6: {
          title: "エンジン/ストリーミングの土台",
          text:
            "C++/CUDAのデータプレーンに加え、ZeroMQやRTP周りの知見も含め、実運用で使える構成を整えます。",
        },
      },
      arch: {
        title: "アーキテクチャ",
        subtitle: "扱いやすい制御系と、速い処理系に分割しています。",
        diagramTitle: "全体像（概要）",
        diagram:
          "Control Plane (Python/FastAPI)     Data Plane (C++ Audio Engine)\n" +
          "├── IR Generator (scipy)           ├── GPU FFT Convolution (CUDA)\n" +
          "├── OPRA Integration               ├── libsoxr Resampling\n" +
          "└── ZeroMQ Command Interface   <-> └── ALSA Output\n",
        roadmapTitle: "ロードマップ（概要）",
        steps: {
          s1: { k: "Phase 1", v: "コアエンジン & ミドルウェア（進行中）" },
          s2: { k: "Phase 2", v: "コントロールプレーン & Web UI" },
          s3: { k: "Phase 3", v: "Jetsonハードウェア統合" },
        },
        note:
          "現在は個人の副業として開発していますが、製品化・ハードウェア開発・DSP/AIの技術支援など、商用化や技術提供は柔軟に対応可能です。",
      },
      contact: {
        title: "お問い合わせ",
        subtitle:
          "製品化の提携、ライセンス/技術支援、PoC/デモ機の試聴、技術相談など、お気軽にご連絡ください。",
        cardTitle: "次の“最高の音”を一緒に作りませんか",
        cardText:
          "下記フォームからお問い合わせください。内容は厳重に管理し、本件の連絡以外には使用しません。",
        form: "フォームを開く",
        copy: "リンクをコピー",
        hint: "ヒント：ポップアップブロックされる場合は、新しいタブで開いてください。",
        copied: "お問い合わせフォームのリンクをコピーしました。",
        copyFailed: "コピーに失敗しました。リンクを手動でコピーしてください。",
      },
      footer: {
        copy:
          "© Totton Audio. このリポジトリ向けのブランドページ初稿。\nブラウザ言語が日本語以外の場合は英語表示がデフォルトです。",
        backToTop: "ページ上部へ",
      },
    },
  };

  function normalizeLang(raw) {
    if (!raw) return "en";
    const s = String(raw).trim().toLowerCase();
    if (s === "ja" || s.startsWith("ja-")) return "ja";
    return "en";
  }

  function detectLang() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("lang");
    if (q) return normalizeLang(q);

    const stored = localStorage.getItem("totton.lang");
    if (stored) return normalizeLang(stored);

    const candidates = []
      .concat(navigator.languages || [])
      .concat([navigator.language])
      .filter(Boolean);
    for (const c of candidates) {
      const nl = normalizeLang(c);
      if (nl === "ja") return "ja";
    }
    return "en";
  }

  function setHtmlLang(lang) {
    try {
      document.documentElement.setAttribute("lang", lang);
    } catch {
      // ignore
    }
  }

  function getDict(lang) {
    const base = DICT.en;
    const pick = DICT[lang] || DICT.en;
    return { base, pick };
  }

  function showToast(ctx, text) {
    ctx.toastText = text;
    ctx.toastOpen = true;
    window.clearTimeout(ctx._toastTimer);
    ctx._toastTimer = window.setTimeout(() => {
      ctx.toastOpen = false;
    }, 2200);
  }

  window.brandPage = function brandPage() {
    const initial = detectLang();
    setHtmlLang(initial);

    const ctx = {
      lang: initial,
      toastOpen: false,
      toastText: "",
      _toastTimer: null,

      init() {
        document.addEventListener("totton:set-lang", (ev) => {
          const next = normalizeLang(ev?.detail?.lang);
          this.setLang(next);
        });
      },

      setLang(next) {
        const nl = normalizeLang(next);
        this.lang = nl;
        localStorage.setItem("totton.lang", nl);
        setHtmlLang(nl);
      },

      t(key) {
        const { base, pick } = getDict(this.lang);
        const parts = String(key).split(".");
        let cur = pick;
        for (const p of parts) cur = cur?.[p];
        if (typeof cur === "string") return cur;
        cur = base;
        for (const p of parts) cur = cur?.[p];
        return typeof cur === "string" ? cur : String(key);
      },

      async copyContactLink() {
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(CONTACT_URL);
          } else {
            // Fallback: selection-based copy
            const ta = document.createElement("textarea");
            ta.value = CONTACT_URL;
            ta.setAttribute("readonly", "true");
            ta.style.position = "fixed";
            ta.style.top = "-9999px";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          showToast(this, this.t("contact.copied"));
        } catch {
          showToast(this, this.t("contact.copyFailed"));
        }
      },
    };

    return ctx;
  };
})();
