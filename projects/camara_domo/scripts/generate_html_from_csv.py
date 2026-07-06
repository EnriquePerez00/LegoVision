import csv
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'reports', 'comparativa_pipelines.csv')
template_path = '/Users/I764690/Code_personal/LegoVision/camara_domo/data/reports/authentic_comparative_report.html'
output_path = '/Users/I764690/Code_personal/LegoVision/camara_domo/reports/current_comparative_report.html'

# Read template
with open(template_path, 'r', encoding='utf-8') as f:
    template_content = f.read()

header = template_content.split('<tbody>')[0] + '<tbody>\n'
footer = '</tbody>' + template_content.split('</tbody>')[1]

# Read CSV
rows_html = ""
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = row.get('tracking_id', 'T000')
        ref_CLASSIC = row.get('ref_CLASSIC', 'Unknown')
        ref_HYBRID = row.get('ref_HYBRID', 'Unknown')
        ref_POSE = row.get('ref_POSE', 'Unknown')
        
        c_CLASSIC = row.get('color_CLASSIC', 'Unknown')
        c_HYBRID = row.get('color_HYBRID', 'Unknown')
        c_POSE = row.get('color_POSE', 'Unknown')
        
        a_CLASSIC = row.get('area_cen_CLASSIC', '0')
        a_HYBRID = row.get('area_cen_HYBRID', '0')
        a_POSE = row.get('area_cen_POSE', '0')
        
        h_CLASSIC = row.get('height_CLASSIC', '0')
        h_HYBRID = row.get('height_HYBRID', '0')
        h_POSE = row.get('height_POSE', '0')
        
        # Build main row
        row_html = f"""
        <tr class="main-row">
            <td rowspan="4" style="border-right: 1px solid var(--border-color); text-align: center; vertical-align: middle;">
                <div style="margin-bottom: 8px;">
                    <img src="crops/{tid}_cen.png" alt="Cenital" class="clickable-img" style="max-width: 80px; max-height: 80px; border-radius: 4px; border: 1px solid var(--border-color);"><br>
                    <small style="color: var(--text-secondary);">Cenital</small>
                </div>
                <div>
                    <img src="crops/{tid}_lat.png" alt="Frontal" class="clickable-img" style="max-width: 80px; max-height: 80px; border-radius: 4px; border: 1px solid var(--border-color);"><br>
                    <small style="color: var(--text-secondary);">Frontal</small>
                </div>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <strong>{tid}</strong><br>
                <div style="margin-top: 8px;">
                    <span class="badge badge-neutral">BBox Cen: N/A</span>
                </div>
                <div style="margin-top: 4px;">
                    <span class="badge badge-neutral">BBox Front: N/A</span>
                </div>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <div style="margin-bottom: 4px;"><strong>Ref:</strong> <span class="badge badge-neutral">Unknown</span></div>
                <div style="margin-bottom: 4px;"><strong>Color:</strong> Unknown</div>
                <div style="margin-bottom: 4px;"><strong>Área:</strong> N/A mm²</div>
                <div><strong>Altura:</strong> N/A mm</div>
            </td>
        </tr>
        """
        
        # CLASSIC
        row_html += f"""
            <tr class="sub-row">
                <td><strong>CLASSIC</strong></td>
                <td><span class="badge badge-neutral">{c_CLASSIC}</span></td>
                <td>{a_CLASSIC} mm²</td>
                <td>{h_CLASSIC} mm</td>
                <td>
                    Ref: <span class="badge badge-neutral">{ref_CLASSIC}</span>
                </td>
            </tr>
        """
        
        # HYBRID
        row_html += f"""
            <tr class="sub-row">
                <td><strong>HYBRID</strong></td>
                <td><span class="badge badge-neutral">{c_HYBRID}</span></td>
                <td>{a_HYBRID} mm²</td>
                <td>{h_HYBRID} mm</td>
                <td>
                    Ref: <span class="badge badge-neutral">{ref_HYBRID}</span>
                </td>
            </tr>
        """
        
        # POSE
        row_html += f"""
            <tr class="sub-row">
                <td><strong>POSE</strong></td>
                <td><span class="badge badge-neutral">{c_POSE}</span></td>
                <td>{a_POSE} mm²</td>
                <td>{h_POSE} mm</td>
                <td>
                    Ref: <span class="badge badge-neutral">{ref_POSE}</span>
                </td>
            </tr>
        """
        
        rows_html += row_html

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(header + rows_html + footer)

print(f"Reporte HTML generado en: {output_path}")
