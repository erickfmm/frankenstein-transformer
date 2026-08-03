# Plan: Renumerar Anexos por Número (eliminar duplicado "Annex B")

## Problema

Hay **dos apéndices con letra "B"** en ambos idiomas:
- `annex-b-recurrent-attention.tex` → `\section{Annex B: ...}` → `\label{sec:annex-transformer}`
- `annex-b-latent-attention.tex` → `\section{Annex B: ...}` → `\label{sec:annex-latent}`

En `paper.tex`/`paper-es.tex` el orden de `\input` es: A→B(recurrent)→B(latent)→C→D→E→F, por lo que el PDF generado muestra dos "Annex B" consecutivos y el índice está roto.

## Solución: Numeración por enteros en lugar de letras

Cambiar de `Annex A, B, C...` a numeración automática `1, 2, 3...` usando `\renewcommand{\thesection}{\arabic{section}}` tras `\appendix`. Las `\label` semánticas (`sec:annex-*`) **no cambian**, así las `\ref` siguen funcionando sin tocar el cuerpo del paper.

## Mapeo de renombrado de archivos

### `docs/paper/appendices/` (inglés)

| Archivo actual | Nuevo nombre | `\section` actual | `\section` nueva |
|---|---|---|---|
| `annex-a-optimizers.tex` | `annex-1-optimizers.tex` | `Annex A: Optimizer Families` | `Optimizer Families` |
| `annex-b-recurrent-attention.tex` | `annex-2-recurrent-attention.tex` | `Annex B: Dense, Recurrent, and Memory-Augmented Transformers` | `Dense, Recurrent, and Memory-Augmented Transformers` |
| `annex-b-latent-attention.tex` | `annex-3-latent-attention.tex` | `Annex B: Latent and Low-Rank Attention Families` | `Latent and Low-Rank Attention Families` |
| `annex-c-sparse-attention.tex` | `annex-4-sparse-attention.tex` | `Annex C: Comprehensive Sparse Attention Mechanisms` | `Comprehensive Sparse Attention Mechanisms` |
| `annex-d-gated-attention.tex` | `annex-5-gated-attention.tex` | `Annex D: Gated Attention Families---Complete Literature Analysis` | `Gated Attention Families---Complete Literature Analysis` |
| `annex-e-tutorial.tex` | `annex-6-tutorial.tex` | `Annex E: Conceptual Introduction---Transformers and Attention for Beginners` | `Conceptual Introduction---Transformers and Attention for Beginners` |
| `annex-f-norm-bitnet-mod.tex` | `annex-7-norm-bitnet-mod.tex` | `Annex F: Normalization, Quantization, Depth Routing, and Runtime Algorithms` | `Normalization, Quantization, Depth Routing, and Runtime Algorithms` |

### `docs/paper-es/appendices/` (español)

| Archivo actual | Nuevo nombre | `\section` actual | `\section` nueva |
|---|---|---|---|
| `annex-a-optimizadores.tex` | `annex-1-optimizadores.tex` | `Anexo A: Familias de Optimizadores` | `Familias de Optimizadores` |
| `annex-b-recurrent-attention.tex` | `annex-2-recurrent-attention.tex` | `Anexo B: Transformadores Densos, Recurrentes y Aumentados con Memoria` | `Transformadores Densos, Recurrentes y Aumentados con Memoria` |
| `annex-b-latent-attention.tex` | `annex-3-latent-attention.tex` | `Anexo B: Familias de Atención Latente y de Rango Bajo` | `Familias de Atención Latente y de Rango Bajo` |
| `annex-c-sparse-attention.tex` | `annex-4-sparse-attention.tex` | `Annex C: Comprehensive Sparse Attention Mechanisms` | `Mecanismos de Atención Dispersa Exhaustivos` |
| `annex-d-gated-attention.tex` | `annex-5-gated-attention.tex` | `Annex D: Gated Attention Families---Complete Literature Analysis` | `Familias de Atención con Compuerta---Análisis Exhaustivo de Literatura` |
| `annex-e-tutorial.tex` | `annex-6-tutorial.tex` | `Annex E: Conceptual Introduction---Transformers and Attention for Beginners` | `Introducción Conceptual---Transformers y Atención para Principiantes` |
| `annex-f-norm-bitnet-mod.tex` | `annex-7-norm-bitnet-mod.tex` | `Anexo F: Normalización, Cuantización, Enrutamiento de Profundidad y Algoritmos de Ejecución` | `Normalización, Cuantización, Enrutamiento de Profundidad y Algoritmos de Ejecución` |

> **Nota**: Los archivos `annex-{c,d,e}-*` en `paper-es/` tienen títulos en inglés (no traducidos). Se traducirán al español en este paso.

## Cambios en archivos principales

### `docs/paper/paper.tex`

```latex
% Línea 67-75: reemplazar bloque \appendix + \input
\appendix
\renewcommand{\thesection}{\arabic{section}}
\addtocontents{toc}{\bigskip\noindent\textbf{Appendices}\par}

\input{appendices/annex-1-optimizers}
\input{appendices/annex-2-recurrent-attention}
\input{appendices/annex-3-latent-attention}
\input{appendices/annex-4-sparse-attention}
\input{appendices/annex-5-gated-attention}
\input{appendices/annex-6-tutorial}
\input{appendices/annex-7-norm-bitnet-mod}
```

### `docs/paper-es/paper-es.tex`

```latex
\appendix
\renewcommand{\thesection}{\arabic{section}}
\addtocontents{toc}{\bigskip\noindent\textbf{Anexos}\par}

\input{appendices/annex-1-optimizadores}
\input{appendices/annex-2-recurrent-attention}
\input{appendices/annex-3-latent-attention}
\input{appendices/annex-4-sparse-attention}
\input{appendices/annex-5-gated-attention}
\input{appendices/annex-6-tutorial}
\input{appendices/annex-7-norm-bitnet-mod}
```

## Cambios dentro de cada archivo de anexo

Para cada archivo de anexo (EN y ES):
1. **Quitar el prefijo manual** "Annex X:" / "Anexo X:" de `\section{...}`
2. **Mantener `\label{sec:annex-*}` intacto** — es semántico, no depende de la letra/número

Ejemplo (EN, `annex-1-optimizers.tex`):
```latex
% Antes:
\section{Annex A: Optimizer Families}
\label{sec:annex-optimizers}

% Después:
\section{Optimizer Families}
\label{sec:annex-optimizers}
```

Ejemplo (ES, `annex-1-optimizadores.tex`):
```latex
% Antes:
\section{Anexo A: Familias de Optimizadores}
\label{sec:annex-optimizers}

% Después:
\section{Familias de Optimizadores}
\label{sec:annex-optimizers}
```

## Lo que NO cambia

- **`\label{sec:annex-*}`**: todas las etiquetas son semánticas (`sec:annex-optimizers`, `sec:annex-transformer`, `sec:annex-latent`, `sec:annex-sparse`, `sec:annex-gated`, `sec:annex-tutorial`, `sec:annex-norm-bitnet-mod`). No dependen de la letra/número del anexo.
- **`\ref{sec:annex-*}`**: las ~15 referencias cruzadas en `sections/*.tex` siguen funcionando. Con la nueva numeración, `\ref{sec:annex-optimizers}` devolverá "1" en lugar de "A". El texto "Appendix~\ref{...}" pasará de leer "Appendix A" a "Appendix 1".
- **`\label{tab:...}`, `\label{alg:...}`, `\label{fig:...}`**: sin cambios.
- **Bibliografía** (`../bibliography/*`): sin cambios.
- **Archivos `.aux`, `.bbl`, `.toc`, etc.**: se regeneran al compilar.

## Referencias cruzadas `\ref` en el cuerpo (verificación — NO requieren cambios)

| Archivo | Texto actual | Texto tras renumerar |
|---|---|---|
| `01-introduction.tex:55` | `Appendix~\ref{sec:annex-tutorial}` | "Appendix 6" |
| `01-introduction.tex:57` | `Appendices~\ref{sec:annex-transformer}, \ref{sec:annex-latent}, \ref{sec:annex-sparse}, \ref{sec:annex-gated}` | "Appendices 2, 3, 4, 5" |
| `01-introduction.tex:57` | `Appendix~\ref{sec:annex-norm-bitnet-mod}` | "Appendix 7" |
| `01-introduction.tex:58` | `Appendix~\ref{sec:annex-optimizers}` | "Appendix 1" |
| `02-related-work.tex:11` | `Appendix~\ref{sec:annex-transformer}` | "Appendix 2" |
| `02-related-work.tex:14` | `Appendix~\ref{sec:annex-sparse}` | "Appendix 4" |
| `02-related-work.tex:17` | `Appendix~\ref{sec:annex-gated}` | "Appendix 5" |
| `02-related-work.tex:31` | `Appendix~\ref{sec:annex-optimizers}` | "Appendix 1" |
| `04-architecture-taxonomy.tex:96` | `Appendix~\ref{sec:annex-transformer}` | "Appendix 2" |
| `04-architecture-taxonomy.tex:99` | `Appendix~\ref{sec:annex-latent}` | "Appendix 3" |
| `04-architecture-taxonomy.tex:102` | `Appendix~\ref{sec:annex-sparse}` | "Appendix 4" |
| `04-architecture-taxonomy.tex:121` | `Appendix~\ref{sec:annex-gated}` | "Appendix 5" |
| `05-optimizer-families.tex:4` | `Appendix~\ref{sec:annex-optimizers}` | "Appendix 1" |
| `06-quantization-deployment.tex:22` | `Appendix~\ref{sec:annex-norm-bitnet-mod}` | "Appendix 7" |
| `07-sbert-tasks.tex:37` | `Appendix~\ref{sec:annex-norm-bitnet-mod}` | "Appendix 7" |
| `03-system-design.tex:271` | `Appendix~\ref{sec:annex-norm-bitnet-mod}` | "Appendix 7" |
| `03-system-design.tex:276` | `Appendix~\ref{sec:annex-norm-bitnet-mod}` | "Appendix 7" |

Todas las referencias cruzadas usan `\ref{sec:annex-*}` (no números hardcodeados), así que funcionan automáticamente tras la renumeración.

## Plan de ejecución paso a paso

### Paso 1: Renombrar archivos (EN)
```bash
cd docs/paper/appendices
git mv annex-a-optimizers.tex       annex-1-optimizers.tex
git mv annex-b-recurrent-attention.tex annex-2-recurrent-attention.tex
git mv annex-b-latent-attention.tex annex-3-latent-attention.tex
git mv annex-c-sparse-attention.tex annex-4-sparse-attention.tex
git mv annex-d-gated-attention.tex  annex-5-gated-attention.tex
git mv annex-e-tutorial.tex        annex-6-tutorial.tex
git mv annex-f-norm-bitnet-mod.tex annex-7-norm-bitnet-mod.tex
```

### Paso 2: Renombrar archivos (ES)
```bash
cd docs/paper-es/appendices
git mv annex-a-optimizadores.tex      annex-1-optimizadores.tex
git mv annex-b-recurrent-attention.tex annex-2-recurrent-attention.tex
git mv annex-b-latent-attention.tex   annex-3-latent-attention.tex
git mv annex-c-sparse-attention.tex   annex-4-sparse-attention.tex
git mv annex-d-gated-attention.tex     annex-5-gated-attention.tex
git mv annex-e-tutorial.tex           annex-6-tutorial.tex
git mv annex-f-norm-bitnet-mod.tex    annex-7-norm-bitnet-mod.tex
```

### Paso 3: Editar `docs/paper/paper.tex`
- Líneas 67-75: añadir `\renewcommand{\thesection}{\arabic{section}}` y `\addtocontents{toc}{...}` tras `\appendix`
- Actualizar los 7 `\input{appendices/...}` con los nuevos nombres

### Paso 4: Editar `docs/paper-es/paper-es.tex`
- Mismos cambios que el paso 3 (con "Anexos" en el TOC)

### Paso 5: Quitar prefijos "Annex X:" / "Anexo X:" de `\section{...}` en los 14 archivos de anexos (7 EN + 7 ES)
- Cada archivo tiene una sola `\section{Annex X: ...}` en la línea 1
- Editar con `edit` tool (o `sed` batch) para quitar "Annex A: " / "Anexo A: " etc.
- Mantener la `\label{sec:annex-*}` de la línea 2 sin cambios

### Paso 6: Traducir títulos no traducidos en `paper-es/appendices/`
Estos 3 archivos en español aún tienen títulos `\section` en inglés:
- `annex-4-sparse-attention.tex`: `Annex C: Comprehensive Sparse Attention Mechanisms` → `Mecanismos de Atención Dispersa Exhaustivos`
- `annex-5-gated-attention.tex`: `Annex D: Gated Attention Families---Complete Literature Analysis` → `Familias de Atención con Compuerta---Análisis Exhaustivo de Literatura`
- `annex-6-tutorial.tex`: `Annex E: Conceptual Introduction---Transformers and Attention for Beginners` → `Introducción Conceptual---Transformers y Atención para Principiantes`

### Paso 7: Limpiar artefactos de compilación obsoletos
```bash
rm -f docs/paper/paper.aux docs/paper/paper.toc docs/paper/paper.bbl docs/paper/paper.blg docs/paper/paper.log docs/paper/paper.out
rm -f docs/paper-es/paper-es.aux docs/paper-es/paper-es.toc docs/paper-es/paper-es.bbl docs/paper-es/paper-es.blg docs/paper-es/paper-es.log docs/paper-es/paper-es.out
```

### Paso 8: Recompilar para verificar
```bash
cd docs/paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
cd docs/paper-es && pdflatex paper-es && bibtex paper-es && pdflatex paper-es && pdflatex paper-es
```
Verificar que:
- No hay warnings de "Label(s) may have changed. Rerun..."
- No hay errores de "undefined reference"
- El índice muestra "Appendices" / "Anexos" como separador y los anexos numerados 1-7 sin duplicados
- Los `\ref` en el cuerpo del texto muestran "Appendix 1", "Appendix 2", etc.

### Paso 9: Actualizar plan histórico (opcional)
`.opencode/plans/cca-ccgqa-implementation.md` línea 114 y 117 referencian `annex-b-latent-attention.tex` (nombre antiguo). Actualizar a `annex-3-latent-attention.tex` para consistencia.

## Verificación de no-regresión

- **Compilar ambos papers sin errores** de LaTeX (undefined references, missing files).
- **Revisar el TOC generado** (`paper.toc` / `paper-es.toc`): debe haber exactamente 7 anexos numerados 1-7, sin saltos ni duplicados.
- **Buscar referencias rotas**: `grep -rn '\\ref{sec:annex' docs/paper/ docs/paper-es/` — todas deben resolver a labels existentes.
- **Buscar archivos huérfanos**: `ls docs/paper/appendices/ docs/paper-es/appendices/` — no debe quedar ningún `annex-{a,b,c,d,e,f}-*`.