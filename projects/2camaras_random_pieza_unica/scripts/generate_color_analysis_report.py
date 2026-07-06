# -*- coding: utf-8 -*-
"""generate_color_analysis_report.py
Análisis de errores de COLOR en la inferencia 300:
estimación CENITAL, LATERAL y FINAL (consenso) vs GT.

Salidas: data/reports/color_analysis.{md,html}
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def hex_to_rgb(h):
    if not h:
        return None
    h = h.lstrip('#')
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return None


def swatch(hex_):
    return f'<span class="sw" style="background:{hex_}"></span>' if hex_ else ''


def delta_html(v):
    if v is None:
        return '-'
    cls = 'd-zero' if abs(v) < 5 else ('d-pos' if v > 0 else 'd-neg')
    return f'<span class="{cls}">{v:+.0f}</span>'


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f8;color:#222;margin:0;padding:24px;max-width:1400px}
h1{margin:0 0 8px;color:#0f172a}
h2{margin-top:24px;color:#1e3a8a;border-bottom:2px solid #cbd5e1;padding-bottom:6px}
h3{color:#1e40af;margin-top:18px}
.sub{color:#64748b;margin-top:0}
.card{background:#fff;border-radius:8px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
th{background:#f1f5f9;font-weight:600}
tr:hover{background:#fafafa}
.bad-cell{background:#fee2e2}
.ok-cell{background:#dcfce7}
.warn-cell{background:#fef3c7}
.sw{display:inline-block;width:18px;height:18px;border:1px solid #cbd5e1;vertical-align:middle;margin-right:6px;border-radius:3px}
.d-pos{color:#b91c1c;font-weight:600}
.d-neg{color:#1e40af;font-weight:600}
.d-zero{color:#64748b}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
.metric{background:#fff;border-radius:8px;padding:12px 16px;border-left:4px solid #2563eb;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.metric.bad{border-left-color:#b91c1c}
.metric.warn{border-left-color:#d97706}
.metric.ok{border-left-color:#15803d}
.metric .v{font-size:24px;font-weight:600;color:#0f172a}
.metric .k{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.metric .extra{color:#475569;font-size:11px;margin-top:4px}
.diag{background:#fff7ed;border-left:4px solid #ea580c;padding:14px 18px;border-radius:6px}
code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-family:ui-monospace,Menlo,monospace}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    # Default: reports/ (separación de dominios)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, 'reports')
    ap.add_argument('--out', default=default_out,
                    help='Directorio de salida para reports (default: reports/)')
    args = ap.parse_args()

    with open(args.eval) as f:
        data = json.load(f)
    results = data.get('results', [])
    universe = [r for r in results if r.get('color_code_gt') is not None]
    n = len(universe)
    if n == 0:
        print('[color] sin muestras')
        return

    # ── Stats ──
    n_cen_bad = sum(1 for r in universe if r.get('color_match_cenital') is False)
    n_lat_bad = sum(1 for r in universe if r.get('color_match_lateral') is False)
    n_cons_bad = sum(1 for r in universe if r.get('color_consensus_ok') is False)
    n_dec_bad = sum(1 for r in universe if r.get('color_decision_used') != r.get('color_code_gt'))
    n_any_bad = sum(1 for r in universe if (r.get('color_match_cenital') is False
                                            or r.get('color_match_lateral') is False
                                            or r.get('color_consensus_ok') is False))

    gt_colors = sorted({r['color_code_gt'] for r in universe},
                       key=lambda c: -sum(1 for r in universe if r['color_code_gt'] == c))
    gt_info = {}
    for c in gt_colors:
        s = next(r for r in universe if r['color_code_gt'] == c)
        gt_info[c] = (s.get('color_name_gt') or '?', s.get('color_hex_gt') or '')

    cen_pred = defaultdict(Counter)
    lat_pred = defaultdict(Counter)
    dec_pred = defaultdict(Counter)
    cen_rgbs = defaultdict(list)
    lat_rgbs = defaultdict(list)
    pred_info = {}  # pred_code -> (name, hex)
    for r in universe:
        gt = r['color_code_gt']
        cn = r.get('color_cenital_normalized_code')
        ln = r.get('color_lateral_normalized_code')
        dn = r.get('color_decision_used')
        cen_pred[gt][cn] += 1
        lat_pred[gt][ln] += 1
        dec_pred[gt][dn] += 1
        if r.get('color_cenital_rgb_est'):
            cen_rgbs[gt].append(r['color_cenital_rgb_est'])
        if r.get('color_lateral_rgb_est'):
            lat_rgbs[gt].append(r['color_lateral_rgb_est'])
        if cn and cn not in pred_info:
            pred_info[cn] = (r.get('color_cenital_normalized_name') or '?', None)
        if ln and ln not in pred_info:
            pred_info[ln] = (r.get('color_lateral_normalized_name') or '?', None)

    os.makedirs(args.out, exist_ok=True)
    md_path = os.path.join(args.out, 'color_analysis.md')
    html_path = os.path.join(args.out, 'color_analysis.html')

    write_markdown(md_path, n, n_cen_bad, n_lat_bad, n_cons_bad, n_dec_bad, n_any_bad,
                   gt_colors, gt_info, cen_pred, lat_pred, dec_pred,
                   cen_rgbs, lat_rgbs, universe, pred_info)
    write_html(html_path, n, n_cen_bad, n_lat_bad, n_cons_bad, n_dec_bad, n_any_bad,
               gt_colors, gt_info, cen_pred, lat_pred, dec_pred,
               cen_rgbs, lat_rgbs, universe, pred_info, args.eval)
    print(f'[color] Markdown: {md_path}')
    print(f'[color] HTML    : {html_path}')


def _name_for(code, gt_info, pred_info):
    if code in gt_info:
        return gt_info[code][0]
    if code in pred_info:
        return pred_info[code][0]
    return '?'


def write_markdown(path, n, n_cen_bad, n_lat_bad, n_cons_bad, n_dec_bad, n_any_bad,
                   gt_colors, gt_info, cen_pred, lat_pred, dec_pred,
                   cen_rgbs, lat_rgbs, universe, pred_info):
    L = []
    L.append('# Análisis de errores de color — Inferencia 300\n')
    L.append(f'- Total muestras: **{n}**')
    L.append(f'- Colores GT distintos: **{len(gt_colors)}** ({", ".join(gt_colors)})\n')
    L.append('## Tasa de fallos por categoría\n')
    L.append('| Categoría | Fallos | % |')
    L.append('|---|---|---|')
    L.append(f'| Color CENITAL erróneo | {n_cen_bad}/{n} | {100*n_cen_bad/n:.1f}% |')
    L.append(f'| Color LATERAL erróneo | {n_lat_bad}/{n} | {100*n_lat_bad/n:.1f}% |')
    L.append(f'| Consenso final NO ok  | {n_cons_bad}/{n} | {100*n_cons_bad/n:.1f}% |')
    L.append(f'| Decisión usada ≠ GT   | {n_dec_bad}/{n} | {100*n_dec_bad/n:.1f}% |')
    L.append(f'| Cualquier mismatch    | {n_any_bad}/{n} | {100*n_any_bad/n:.1f}% |\n')

    L.append('## Aciertos por color GT\n')
    L.append('| GT | nombre | hex | n | cen ✓ | lat ✓ | decisión ✓ |')
    L.append('|---|---|---|---|---|---|---|')
    for c in gt_colors:
        name, hex_ = gt_info[c]
        sub = [r for r in universe if r['color_code_gt'] == c]
        nn = len(sub)
        co = sum(1 for r in sub if r.get('color_match_cenital'))
        lo = sum(1 for r in sub if r.get('color_match_lateral'))
        do = sum(1 for r in sub if r.get('color_decision_used') == c)
        L.append(f'| {c} | {name} | {hex_} | {nn} | {co}/{nn} ({100*co/nn:.0f}%) | {lo}/{nn} ({100*lo/nn:.0f}%) | {do}/{nn} ({100*do/nn:.0f}%) |')
    L.append('')

    L.append('## RGB esperado vs medido (CENITAL)\n')
    L.append('| GT | nombre | hex GT | RGB esperado | RGB medido (media) | ΔR | ΔG | ΔB | n |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for c in gt_colors:
        name, hex_ = gt_info[c]
        rgbs = cen_rgbs[c]
        if not rgbs:
            continue
        gt_rgb = hex_to_rgb(hex_) or (0, 0, 0)
        R = statistics.mean(x[0] for x in rgbs)
        G = statistics.mean(x[1] for x in rgbs)
        B = statistics.mean(x[2] for x in rgbs)
        L.append(f'| {c} | {name} | `{hex_}` | ({gt_rgb[0]},{gt_rgb[1]},{gt_rgb[2]}) | ({R:.0f},{G:.0f},{B:.0f}) | {R-gt_rgb[0]:+.0f} | {G-gt_rgb[1]:+.0f} | {B-gt_rgb[2]:+.0f} | {len(rgbs)} |')
    L.append('')

    L.append('## RGB esperado vs medido (LATERAL)\n')
    L.append('| GT | nombre | hex GT | RGB esperado | RGB medido (media) | ΔR | ΔG | ΔB | n |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for c in gt_colors:
        name, hex_ = gt_info[c]
        rgbs = lat_rgbs[c]
        if not rgbs:
            continue
        gt_rgb = hex_to_rgb(hex_) or (0, 0, 0)
        R = statistics.mean(x[0] for x in rgbs)
        G = statistics.mean(x[1] for x in rgbs)
        B = statistics.mean(x[2] for x in rgbs)
        L.append(f'| {c} | {name} | `{hex_}` | ({gt_rgb[0]},{gt_rgb[1]},{gt_rgb[2]}) | ({R:.0f},{G:.0f},{B:.0f}) | {R-gt_rgb[0]:+.0f} | {G-gt_rgb[1]:+.0f} | {B-gt_rgb[2]:+.0f} | {len(rgbs)} |')
    L.append('')

    # Confusion matrices
    all_pred_codes = set()
    for d in (cen_pred, lat_pred, dec_pred):
        for v in d.values():
            all_pred_codes.update(k for k in v.keys() if k is not None)
    pred_codes = sorted(all_pred_codes,
                        key=lambda c: -sum(cen_pred[g].get(c, 0) + lat_pred[g].get(c, 0) for g in gt_colors))

    def _conf_md(title, mat):
        L.append(f'## Matriz de confusión — {title}\n')
        L.append('| GT \\ Pred | ' + ' | '.join(p for p in pred_codes) + ' | otros |')
        L.append('|---' + ('|---' * (len(pred_codes) + 1)) + '|')
        for gt in gt_colors:
            row = [f'{gt} ({gt_info[gt][0]})']
            for p in pred_codes:
                v = mat[gt].get(p, 0)
                row.append(f'**{v}**' if (v and gt == p) else (str(v) if v else ''))
            otros = sum(c for k, c in mat[gt].items() if k not in pred_codes and k is not None)
            row.append(str(otros) if otros else '')
            L.append('| ' + ' | '.join(row) + ' |')
        L.append('')

    _conf_md('CENITAL', cen_pred)
    _conf_md('LATERAL', lat_pred)
    _conf_md('DECISIÓN FINAL (color_decision_used)', dec_pred)

    L.append('## 🔎 Diagnóstico de la causa raíz\n')
    L.append('El RGB **medido** difiere sistemáticamente del RGB **esperado del catálogo**.\n')
    L.append('### Patrones observados\n')
    L.append('1. **Negro (#202020) → gris azulado (~89,100,105)**: la pieza no se ve negra, su iluminación de fondo la satura.')
    L.append('2. **Blanco (#F9F9F9) → gris (~156,165,170)**: el blanco no llega a saturar; sale gris claro tirando a azul.')
    L.append('3. **Rojo (#C30025) → rosa-rojo apagado (~197,92,122)**: el azul se duplica respecto al GT (37→122), aclarando el rojo.')
    L.append('4. **Dark Bluish Gray (#6B6D67) → cae sobre Light Bluish Gray (~131,142,146)**: confusión por sobreiluminación.')
    L.append('')
    L.append('### Causa raíz probable: iluminación de estudio "lab_lightbox"\n')
    L.append('La función `setup_lab_lightbox()` (ver `scripts/generate_test_set.py:137`) crea:')
    L.append('- 1× Dome cenital de **2000 W**, 35×35')
    L.append('- 4× paredes laterales de **600 W** c/u')
    L.append('- 1× ground fill de **200 W**')
    L.append('- World background blanco @ strength 0.3')
    L.append('')
    L.append('Total ≈ **4600 W de luz blanca difusa**, que produce dos efectos:')
    L.append('- Lava los colores oscuros (Black sale gris)')
    L.append('- Sobreexpone los colores saturados (Red pierde saturación → tirando a rosa)')
    L.append('- El **fondo gris de la cinta + paredes blancas** introduce un sesgo gris-azulado uniforme en todas las piezas → *todo* tiende hacia "Light Bluish Gray".')
    L.append('')
    L.append('### Recomendaciones\n')
    L.append('1. **Reducir la energía total** (Dome a 800 W, paredes a 200 W, ground fill a 50 W) o convertir alguna pared en negativa.')
    L.append('2. **Calibrar el matcher de color** para que aplique la transformación inversa de la iluminación (white-point correction):')
    L.append('   - Capturar el RGB del fondo (cinta) en cada render → estimar offset')
    L.append('   - Restar el offset al RGB de la pieza antes de hacer `find_closest_catalog_color`.')
    L.append('3. **Color management**: forzar `scene.view_settings.view_transform = "Standard"` (no Filmic) para que los hex del catálogo se mapeen 1:1 al render.')
    L.append('4. **Hacer renders del catálogo de color como referencia** (un cubo neutro por color) y normalizar usando esa LUT en lugar de los hex teóricos.')
    L.append('')
    L.append('### Lo que NO es la causa\n')
    L.append('- ❌ La SAM/segmentación: la inferencia de color sí coge la pieza correcta (verificado en sample 6 Red, RGB medido en pieza ≈ RGB reportado).')
    L.append('- ❌ El bbox YOLO: las confianzas son altas (>0.7) y la pieza se localiza bien.')
    L.append('- ❌ El catálogo o el `find_closest_catalog_color`: dado un RGB observado erróneo, hace lo correcto al asignar el color más cercano.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def write_html(path, n, n_cen_bad, n_lat_bad, n_cons_bad, n_dec_bad, n_any_bad,
               gt_colors, gt_info, cen_pred, lat_pred, dec_pred,
               cen_rgbs, lat_rgbs, universe, pred_info, eval_path):
    out = ['<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">'
           '<title>Análisis de color — Inferencia 300</title>'
           f'<style>{CSS}</style></head><body>']
    out.append('<h1>Análisis de errores de color — Inferencia 300</h1>')
    out.append(f'<p class="sub">Total muestras: <b>{n}</b> · Colores GT: <b>{len(gt_colors)}</b> · '
               f'Eval: <code>{os.path.basename(eval_path)}</code></p>')

    def _cls(p):
        return 'bad' if p > 30 else ('warn' if p > 10 else 'ok')

    out.append('<div class="grid">')
    for label, val, total in [
        ('Mismatch CENITAL', n_cen_bad, n),
        ('Mismatch LATERAL', n_lat_bad, n),
        ('Fallo consenso', n_cons_bad, n),
        ('Decisión ≠ GT', n_dec_bad, n),
        ('Cualquier mismatch', n_any_bad, n),
    ]:
        pct = 100 * val / total
        out.append(f'<div class="metric {_cls(pct)}"><div class="k">{label}</div>'
                   f'<div class="v">{val}/{total}</div><div class="extra">{pct:.1f}%</div></div>')
    out.append('</div>')

    # Aciertos por color
    out.append('<div class="card"><h2>Aciertos por color GT</h2>')
    out.append('<table><thead><tr><th>GT</th><th>nombre</th><th>hex</th><th>n</th>'
               '<th>cenital ✓</th><th>lateral ✓</th><th>decisión final ✓</th></tr></thead><tbody>')
    for c in gt_colors:
        name, hex_ = gt_info[c]
        sub = [r for r in universe if r['color_code_gt'] == c]
        nn = len(sub)
        co = sum(1 for r in sub if r.get('color_match_cenital'))
        lo = sum(1 for r in sub if r.get('color_match_lateral'))
        do = sum(1 for r in sub if r.get('color_decision_used') == c)

        def _cell(ok, total):
            pct = ok / total if total else 0
            cls = 'ok-cell' if pct > 0.8 else ('bad-cell' if pct < 0.2 else 'warn-cell')
            return f'<td class="{cls}">{ok}/{total} ({100*pct:.0f}%)</td>'

        out.append(f'<tr><td>{c}</td><td>{swatch(hex_)}{name}</td>'
                   f'<td><code>{hex_}</code></td><td>{nn}</td>'
                   f'{_cell(co, nn)}{_cell(lo, nn)}{_cell(do, nn)}</tr>')
    out.append('</tbody></table></div>')

    # RGB esperado vs medido
    def _rgb_table(title, rgbs_dict):
        out.append(f'<div class="card"><h2>RGB esperado vs medido — {title}</h2>')
        out.append('<table><thead><tr><th>GT</th><th>nombre</th><th>hex GT</th>'
                   '<th>RGB esperado</th><th>RGB medido (media)</th>'
                   '<th>ΔR</th><th>ΔG</th><th>ΔB</th><th>n</th></tr></thead><tbody>')
        for c in gt_colors:
            name, hex_ = gt_info[c]
            rgbs = rgbs_dict[c]
            if not rgbs:
                continue
            gt_rgb = hex_to_rgb(hex_) or (0, 0, 0)
            R = statistics.mean(x[0] for x in rgbs)
            G = statistics.mean(x[1] for x in rgbs)
            B = statistics.mean(x[2] for x in rgbs)
            measured_hex = '#{:02X}{:02X}{:02X}'.format(
                max(0, min(255, int(round(R)))),
                max(0, min(255, int(round(G)))),
                max(0, min(255, int(round(B))))
            )
            out.append(
                f'<tr><td>{c}</td><td>{swatch(hex_)}{name}</td>'
                f'<td><code>{hex_}</code></td>'
                f'<td>({gt_rgb[0]}, {gt_rgb[1]}, {gt_rgb[2]})</td>'
                f'<td>{swatch(measured_hex)}({R:.0f}, {G:.0f}, {B:.0f})</td>'
                f'<td>{delta_html(R-gt_rgb[0])}</td>'
                f'<td>{delta_html(G-gt_rgb[1])}</td>'
                f'<td>{delta_html(B-gt_rgb[2])}</td>'
                f'<td>{len(rgbs)}</td></tr>'
            )
        out.append('</tbody></table></div>')

    _rgb_table('CENITAL', cen_rgbs)
    _rgb_table('LATERAL', lat_rgbs)

    # Confusion matrices
    all_pred_codes = set()
    for d in (cen_pred, lat_pred, dec_pred):
        for v in d.values():
            all_pred_codes.update(k for k in v.keys() if k is not None)
    pred_codes = sorted(all_pred_codes,
                        key=lambda c: -sum(cen_pred[g].get(c, 0) + lat_pred[g].get(c, 0) for g in gt_colors))

    def _conf_html(title, mat):
        out.append(f'<div class="card"><h2>Matriz de confusión — {title}</h2>')
        out.append('<table><thead><tr><th>GT \\ Pred</th>')
        for p in pred_codes:
            pname = _name_for(p, gt_info, pred_info)
            out.append(f'<th title="{pname}">{p}<br/><span class="sub" style="font-size:10px">{pname[:14]}</span></th>')
        out.append('<th>otros</th><th>total</th></tr></thead><tbody>')
        for gt in gt_colors:
            row_total = sum(mat[gt].values())
            out.append(f'<tr><td><b>{gt}</b><br/>{swatch(gt_info[gt][1])}{gt_info[gt][0]}</td>')
            for p in pred_codes:
                v = mat[gt].get(p, 0)
                if v == 0:
                    out.append('<td></td>')
                else:
                    cls = 'ok-cell' if gt == p else ('bad-cell' if v > row_total*0.4 else 'warn-cell')
                    out.append(f'<td class="{cls}">{v}</td>')
            otros = sum(c for k, c in mat[gt].items() if k not in pred_codes and k is not None)
            out.append(f'<td>{otros if otros else ""}</td><td>{row_total}</td></tr>')
        out.append('</tbody></table></div>')

    _conf_html('CENITAL', cen_pred)
    _conf_html('LATERAL', lat_pred)
    _conf_html('DECISIÓN FINAL (color_decision_used)', dec_pred)

    # Diagnóstico
    out.append('<div class="card"><h2>🔎 Diagnóstico de la causa raíz</h2>')
    out.append('<div class="diag">'
               '<p>El RGB <b>medido</b> difiere sistemáticamente del RGB <b>esperado</b> del catálogo. '
               'Los patrones observados son consistentes con un <b>sesgo sistemático de iluminación</b>:</p>'
               '<ul>'
               '<li><b>Negro</b> (#202020) → gris azulado (~89,100,105). La pieza no llega a verse oscura.</li>'
               '<li><b>Blanco</b> (#F9F9F9) → gris claro (~156,165,170). El blanco no satura.</li>'
               '<li><b>Rojo</b> (#C30025) → rosa-rojo apagado (~197,92,122). El canal B sube de 37 a 122.</li>'
               '<li><b>Dark Bluish Gray</b> (#6B6D67) → confundido con Light Bluish Gray.</li>'
               '</ul>'
               '<h3>Causa raíz probable</h3>'
               '<p>La función <code>setup_lab_lightbox()</code> '
               '(<code>scripts/generate_test_set.py:137</code>) configura una caja de luz con:</p>'
               '<ul>'
               '<li>1× Dome cenital de <b>2000 W</b> (35×35)</li>'
               '<li>4× paredes laterales de <b>600 W</b> cada una</li>'
               '<li>1× ground fill de <b>200 W</b></li>'
               '<li>World background blanco @ strength 0.3</li>'
               '</ul>'
               '<p>≈ <b>4 600 W de luz blanca difusa</b>: lava los oscuros, satura los rojos y proyecta '
               'el gris-azulado del fondo (cinta + paredes) sobre cada pieza, sesgando todo hacia '
               '<i>Light Bluish Gray</i>.</p>'
               '<h3>Recomendaciones</h3>'
               '<ol>'
               '<li>Reducir la energía total (Dome 800 W, paredes 200 W, ground 50 W).</li>'
               '<li>Aplicar <i>white-point correction</i>: medir el RGB del fondo y restarlo al RGB '
               'estimado de la pieza antes del matching.</li>'
               '<li>Forzar <code>scene.view_settings.view_transform = "Standard"</code> en lugar de Filmic.</li>'
               '<li>Generar un cubo neutro por color de catálogo y usar esa LUT como referencia '
               'en lugar de los hex teóricos.</li>'
               '</ol>'
               '<h3>Lo que NO es la causa</h3>'
               '<ul>'
               '<li>❌ La SAM/segmentación funciona: el RGB en la zona de la pieza coincide con el reportado.</li>'
               '<li>❌ El bbox YOLO (confianzas &gt; 0.7).</li>'
               '<li>❌ El catálogo o <code>find_closest_catalog_color</code>: dado un RGB sesgado, '
               'devuelve correctamente el color más cercano.</li>'
               '</ul>'
               '</div></div>')
    out.append('</body></html>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


if __name__ == '__main__':
    main()
