"""
Clinical Analysis Module
Generates clinical interpretations and recommendations
"""

import numpy as np
from typing import Dict, List, Any

def generate_clinical_findings(analysis_results: Dict[str, Any]) -> List[str]:
    """
    Generate simple clinical findings list from analysis results
    
    Parameters:
    -----------
    analysis_results : dict
        ECG analysis results
    
    Returns:
    --------
    findings : list
        List of clinical finding strings
    """
    findings = []
    
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    normal_pct = analysis_results.get('normal_percentage', 100)
    hrv_metrics = analysis_results.get('hrv_metrics', {})
    sdnn = hrv_metrics.get('sdnn', 50)
    
    # Heart rate findings
    if hr > 100:
        findings.append(f"Tachycardia detected: Heart rate of {hr} bpm")
    elif hr < 60 and hr > 0:
        findings.append(f"Bradycardia detected: Heart rate of {hr} bpm")
    else:
        findings.append(f"Normal heart rate: {hr} bpm")
    
    # PVC findings
    if pvc_burden > 10:
        findings.append(f"Significant PVC burden: {pvc_burden:.1f}%")
    elif pvc_burden > 5:
        findings.append(f"Moderate PVC burden: {pvc_burden:.1f}%")
    elif pvc_burden > 0:
        findings.append(f"Occasional PVCs detected: {pvc_burden:.1f}%")
    
    # SVE findings
    if sve_burden > 10:
        findings.append(f"Frequent SVE burden: {sve_burden:.1f}%")
    elif sve_burden > 5:
        findings.append(f"Moderate SVE burden: {sve_burden:.1f}%")
    elif sve_burden > 0:
        findings.append(f"Occasional SVEs detected: {sve_burden:.1f}%")
    
    # HRV findings
    if sdnn < 50:
        findings.append(f"Reduced heart rate variability (SDNN: {sdnn:.1f}ms)")
    else:
        findings.append(f"Normal heart rate variability (SDNN: {sdnn:.1f}ms)")
    
    return findings

def generate_clinical_report(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive clinical report based on ECG analysis
    
    Parameters:
    -----------
    analysis_results : dict
        Results from ECG analysis
    
    Returns:
    --------
    report : dict
        Clinical report with findings and recommendations
    """
    report = {
        'summary': '',
        'findings': [],
        'recommendations': [],
        'risk_level': 'low',  # low, moderate, high, critical
        'follow_up': 'routine',  # routine, soon, urgent, immediate
        'interpretation': ''
    }
    
    # Extract key metrics
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    normal_pct = analysis_results.get('normal_percentage', 100)
    hrv = analysis_results.get('hrv_metrics', {})
    signal_quality = analysis_results.get('signal_quality_score', 0)
    
    # Analyze rhythm
    rhythm_interpretation = analyze_rhythm(hr, normal_pct)
    report['findings'].append(rhythm_interpretation)
    
    # Analyze arrhythmia burden
    arrhythmia_findings = analyze_arrhythmia_burden(pvc_burden, sve_burden, normal_pct)
    report['findings'].extend(arrhythmia_findings)
    
    # Analyze HRV
    hrv_findings = analyze_hrv(hrv)
    if hrv_findings:
        report['findings'].append(hrv_findings)
    
    # Signal quality assessment
    if signal_quality < 0.5:
        report['findings'].append({
            'type': 'quality',
            'severity': 'warning',
            'description': 'Poor signal quality detected. Results should be interpreted with caution.',
            'details': f'Signal quality score: {signal_quality:.2f}'
        })
    
    # Determine risk level and recommendations
    report['risk_level'] = determine_risk_level(pvc_burden, sve_burden, normal_pct, hr, hrv)
    report['recommendations'] = generate_recommendations(report['risk_level'], 
                                                        pvc_burden, sve_burden, hr, hrv)
    report['follow_up'] = determine_follow_up(report['risk_level'])
    
    # Generate summary
    report['summary'] = generate_summary(report)
    
    # Overall interpretation
    report['interpretation'] = generate_interpretation(analysis_results, report)
    
    return report

def analyze_rhythm(hr: int, normal_pct: float) -> Dict[str, Any]:
    """
    Analyze heart rhythm characteristics
    """
    finding = {
        'type': 'rhythm',
        'severity': 'normal',
        'description': '',
        'details': ''
    }
    
    if hr == 0:
        finding['severity'] = 'error'
        finding['description'] = 'Unable to determine heart rate'
        return finding
    
    # Heart rate analysis
    if hr < 60:
        finding['severity'] = 'moderate'
        finding['description'] = f'Bradycardia detected'
        finding['details'] = f'Heart rate of {hr} bpm is below normal range (60-100 bpm)'
    elif hr > 100:
        finding['severity'] = 'moderate'
        finding['description'] = f'Tachycardia detected'
        finding['details'] = f'Heart rate of {hr} bpm is above normal range (60-100 bpm)'
    else:
        finding['severity'] = 'normal'
        finding['description'] = 'Normal heart rate'
        finding['details'] = f'Heart rate of {hr} bpm is within normal range (60-100 bpm)'
    
    # Rhythm regularity
    if normal_pct > 95:
        finding['description'] += ' with regular rhythm'
    elif normal_pct > 80:
        finding['description'] += ' with occasional irregularities'
    else:
        finding['severity'] = 'high'
        finding['description'] += ' with frequent irregularities'
    
    return finding

def analyze_arrhythmia_burden(pvc_burden: float, sve_burden: float, 
                              normal_pct: float) -> List[Dict[str, Any]]:
    """
    Analyze arrhythmia burden and significance
    """
    findings = []
    
    # PVC analysis
    if pvc_burden > 0:
        pvc_finding = {
            'type': 'arrhythmia',
            'subtype': 'PVC',
            'severity': 'normal',
            'description': '',
            'details': ''
        }
        
        if pvc_burden > 20:
            pvc_finding['severity'] = 'high'
            pvc_finding['description'] = 'Very frequent premature ventricular contractions'
            pvc_finding['details'] = f'PVC burden: {pvc_burden:.1f}% (>20% is considered very high)'
        elif pvc_burden > 10:
            pvc_finding['severity'] = 'moderate'
            pvc_finding['description'] = 'Frequent premature ventricular contractions'
            pvc_finding['details'] = f'PVC burden: {pvc_burden:.1f}% (>10% is considered significant)'
        elif pvc_burden > 5:
            pvc_finding['severity'] = 'mild'
            pvc_finding['description'] = 'Moderate PVC burden'
            pvc_finding['details'] = f'PVC burden: {pvc_burden:.1f}% (5-10% is moderate)'
        else:
            pvc_finding['severity'] = 'normal'
            pvc_finding['description'] = 'Occasional PVCs detected'
            pvc_finding['details'] = f'PVC burden: {pvc_burden:.1f}% (<5% is generally benign)'
        
        findings.append(pvc_finding)
    
    # SVE analysis
    if sve_burden > 0:
        sve_finding = {
            'type': 'arrhythmia',
            'subtype': 'SVE',
            'severity': 'normal',
            'description': '',
            'details': ''
        }
        
        if sve_burden > 10:
            sve_finding['severity'] = 'moderate'
            sve_finding['description'] = 'Frequent supraventricular ectopy'
            sve_finding['details'] = f'SVE burden: {sve_burden:.1f}% (>10% is considered frequent)'
        elif sve_burden > 5:
            sve_finding['severity'] = 'mild'
            sve_finding['description'] = 'Moderate supraventricular ectopy'
            sve_finding['details'] = f'SVE burden: {sve_burden:.1f}% (5-10% is moderate)'
        else:
            sve_finding['severity'] = 'normal'
            sve_finding['description'] = 'Occasional SVEs detected'
            sve_finding['details'] = f'SVE burden: {sve_burden:.1f}% (<5% is generally benign)'
        
        findings.append(sve_finding)
    
    return findings

def analyze_hrv(hrv_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze heart rate variability metrics
    """
    if not hrv_metrics:
        return None
    
    sdnn = hrv_metrics.get('sdnn', 0)
    rmssd = hrv_metrics.get('rmssd', 0)
    pnn50 = hrv_metrics.get('pnn50', 0)
    
    finding = {
        'type': 'hrv',
        'severity': 'normal',
        'description': '',
        'details': ''
    }
    
    # SDNN analysis (normal: 50-100ms)
    if sdnn < 30:
        finding['severity'] = 'high'
        finding['description'] = 'Severely reduced heart rate variability'
        finding['details'] = f'SDNN: {sdnn:.1f}ms (normal: 50-100ms). May indicate autonomic dysfunction'
    elif sdnn < 50:
        finding['severity'] = 'moderate'
        finding['description'] = 'Reduced heart rate variability'
        finding['details'] = f'SDNN: {sdnn:.1f}ms (normal: 50-100ms). May indicate autonomic imbalance'
    elif sdnn > 150:
        finding['severity'] = 'mild'
        finding['description'] = 'Increased heart rate variability'
        finding['details'] = f'SDNN: {sdnn:.1f}ms. May indicate good autonomic function or arrhythmias'
    else:
        finding['severity'] = 'normal'
        finding['description'] = 'Normal heart rate variability'
        finding['details'] = f'SDNN: {sdnn:.1f}ms, RMSSD: {rmssd:.1f}ms, pNN50: {pnn50:.1f}%'
    
    return finding

def determine_risk_level(pvc_burden: float, sve_burden: float, 
                         normal_pct: float, hr: int, 
                         hrv: Dict[str, float]) -> str:
    """
    Determine overall risk level based on findings
    """
    risk_score = 0
    
    # PVC burden contribution
    if pvc_burden > 20:
        risk_score += 3
    elif pvc_burden > 10:
        risk_score += 2
    elif pvc_burden > 5:
        risk_score += 1
    
    # SVE burden contribution
    if sve_burden > 10:
        risk_score += 2
    elif sve_burden > 5:
        risk_score += 1
    
    # Heart rate contribution
    if hr < 40 or hr > 120:
        risk_score += 3
    elif hr < 50 or hr > 100:
        risk_score += 1
    
    # Rhythm regularity
    if normal_pct < 70:
        risk_score += 3
    elif normal_pct < 85:
        risk_score += 2
    elif normal_pct < 95:
        risk_score += 1
    
    # HRV contribution
    sdnn = hrv.get('sdnn', 50)
    if sdnn < 30:
        risk_score += 2
    elif sdnn < 50:
        risk_score += 1
    
    # Determine risk level
    if risk_score >= 8:
        return 'critical'
    elif risk_score >= 5:
        return 'high'
    elif risk_score >= 3:
        return 'moderate'
    else:
        return 'low'

def generate_recommendations(risk_level: str, pvc_burden: float, 
                            sve_burden: float, hr: int, 
                            hrv: Dict[str, float]) -> List[str]:
    """
    Generate clinical recommendations based on findings
    """
    recommendations = []
    
    # Risk-based recommendations
    if risk_level == 'critical':
        recommendations.append('Immediate medical attention recommended')
        recommendations.append('Consider emergency department evaluation')
    elif risk_level == 'high':
        recommendations.append('Prompt cardiology consultation recommended')
        recommendations.append('24-hour Holter monitoring advised')
        recommendations.append('Consider echocardiogram for structural evaluation')
    elif risk_level == 'moderate':
        recommendations.append('Follow-up with primary care physician')
        recommendations.append('Consider cardiology referral if symptoms persist')
    else:
        recommendations.append('Continue routine cardiac care')
        recommendations.append('Maintain healthy lifestyle habits')
    
    # Specific recommendations based on findings
    if pvc_burden > 10:
        recommendations.append('Evaluate for underlying cardiac conditions')
        recommendations.append('Consider electrolyte panel and thyroid function tests')
        recommendations.append('Avoid stimulants (caffeine, alcohol, nicotine)')
    
    if sve_burden > 10:
        recommendations.append('Monitor for atrial fibrillation development')
        recommendations.append('Consider event monitor for symptom correlation')
    
    if hr < 50:
        recommendations.append('Evaluate for symptomatic bradycardia')
        recommendations.append('Review medications for bradycardic effects')
    elif hr > 100:
        recommendations.append('Investigate underlying causes of tachycardia')
        recommendations.append('Consider beta-blocker therapy if appropriate')
    
    sdnn = hrv.get('sdnn', 50)
    if sdnn < 50:
        recommendations.append('Stress reduction techniques may be beneficial')
        recommendations.append('Regular moderate exercise recommended')
    
    return recommendations

def determine_follow_up(risk_level: str) -> str:
    """
    Determine appropriate follow-up timing
    """
    follow_up_map = {
        'critical': 'immediate',
        'high': 'urgent',
        'moderate': 'soon',
        'low': 'routine'
    }
    return follow_up_map.get(risk_level, 'routine')

def generate_summary(report: Dict[str, Any]) -> str:
    """
    Generate executive summary of findings
    """
    risk_level = report.get('risk_level', 'unknown')
    findings = report.get('findings', [])
    
    # Count significant findings
    significant_findings = [f for f in findings if f.get('severity') in ['moderate', 'high', 'critical']]
    
    if not significant_findings:
        summary = "ECG analysis reveals predominantly normal cardiac rhythm with no significant abnormalities detected."
    else:
        summary = f"ECG analysis reveals {len(significant_findings)} significant finding(s). "
        
        # Summarize main findings
        main_issues = []
        for finding in significant_findings:
            if finding.get('type') == 'rhythm':
                main_issues.append(finding.get('description', ''))
            elif finding.get('type') == 'arrhythmia':
                main_issues.append(finding.get('description', ''))
        
        if main_issues:
            summary += "Key findings include: " + ", ".join(main_issues[:3]) + "."
    
    # Add risk assessment
    risk_descriptions = {
        'low': ' Overall cardiac risk appears low.',
        'moderate': ' Overall cardiac risk is moderate, warranting follow-up.',
        'high': ' Overall cardiac risk is elevated, requiring prompt evaluation.',
        'critical': ' Critical findings present, requiring immediate medical attention.'
    }
    summary += risk_descriptions.get(risk_level, '')
    
    return summary

def generate_interpretation(analysis_results: Dict[str, Any], 
                           report: Dict[str, Any]) -> str:
    """
    Generate detailed clinical interpretation
    """
    interpretation = []
    
    # Opening statement
    total_beats = analysis_results.get('total_beats', 0)
    signal_duration = total_beats / (analysis_results.get('avg_heart_rate', 60) / 60)
    interpretation.append(f"Analysis of {total_beats} cardiac cycles over approximately {signal_duration:.1f} seconds.")
    
    # Rhythm interpretation
    hr = analysis_results.get('avg_heart_rate', 0)
    normal_pct = analysis_results.get('normal_percentage', 0)
    
    if normal_pct > 95:
        rhythm_desc = "predominantly regular sinus rhythm"
    elif normal_pct > 80:
        rhythm_desc = "sinus rhythm with occasional ectopic beats"
    elif normal_pct > 60:
        rhythm_desc = "irregular rhythm with frequent ectopic activity"
    else:
        rhythm_desc = "markedly irregular rhythm"
    
    interpretation.append(f"The recording demonstrates {rhythm_desc} with an average heart rate of {hr} bpm.")
    
    # Ectopic activity
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    
    if pvc_burden > 0 or sve_burden > 0:
        ectopic_desc = []
        if pvc_burden > 0:
            ectopic_desc.append(f"ventricular ectopy (PVC burden: {pvc_burden:.1f}%)")
        if sve_burden > 0:
            ectopic_desc.append(f"supraventricular ectopy (SVE burden: {sve_burden:.1f}%)")
        
        interpretation.append(f"Ectopic activity includes {' and '.join(ectopic_desc)}.")
    
    # HRV assessment
    hrv = analysis_results.get('hrv_metrics', {})
    if hrv:
        sdnn = hrv.get('sdnn', 0)
        if sdnn > 0:
            if sdnn < 50:
                hrv_desc = "reduced heart rate variability suggesting autonomic dysfunction"
            elif sdnn > 100:
                hrv_desc = "increased heart rate variability"
            else:
                hrv_desc = "normal heart rate variability"
            
            interpretation.append(f"Heart rate variability analysis shows {hrv_desc} (SDNN: {sdnn:.1f}ms).")
    
    # Clinical significance
    risk_level = report.get('risk_level', 'unknown')
    if risk_level == 'low':
        interpretation.append("These findings are likely benign and do not suggest significant cardiac pathology.")
    elif risk_level == 'moderate':
        interpretation.append("These findings warrant clinical correlation and may benefit from further cardiac evaluation.")
    elif risk_level == 'high':
        interpretation.append("These findings are clinically significant and warrant prompt cardiac evaluation.")
    elif risk_level == 'critical':
        interpretation.append("These findings are concerning for significant cardiac pathology and require immediate medical attention.")
    
    return " ".join(interpretation)

def calculate_arrhythmia_statistics(beat_classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate detailed arrhythmia statistics
    """
    stats = {
        'total_beats': len(beat_classifications),
        'beat_counts': {},
        'beat_percentages': {},
        'consecutive_abnormal': 0,
        'max_consecutive_abnormal': 0,
        'bigeminy_episodes': 0,
        'trigeminy_episodes': 0,
        'couplets': 0,
        'triplets': 0
    }
    
    # Count beat types
    for beat in beat_classifications:
        beat_type = beat.get('label', 'Unknown')
        stats['beat_counts'][beat_type] = stats['beat_counts'].get(beat_type, 0) + 1
    
    # Calculate percentages
    total = stats['total_beats']
    if total > 0:
        for beat_type, count in stats['beat_counts'].items():
            stats['beat_percentages'][beat_type] = (count / total) * 100
    
    # Analyze patterns
    consecutive_pvc = 0
    for i, beat in enumerate(beat_classifications):
        if beat.get('label') == 'PVC':
            consecutive_pvc += 1
            stats['max_consecutive_abnormal'] = max(stats['max_consecutive_abnormal'], consecutive_pvc)
            
            # Check for couplets (2 consecutive PVCs)
            if consecutive_pvc == 2:
                stats['couplets'] += 1
            # Check for triplets (3 consecutive PVCs)
            elif consecutive_pvc == 3:
                stats['triplets'] += 1
        else:
            consecutive_pvc = 0
        
        # Check for bigeminy (alternating normal and PVC)
        if i > 0 and i < len(beat_classifications) - 1:
            if (beat_classifications[i-1].get('label') == 'Normal' and 
                beat.get('label') == 'PVC' and 
                beat_classifications[i+1].get('label') == 'Normal'):
                stats['bigeminy_episodes'] += 1
        
        # Check for trigeminy (2 normal, 1 PVC pattern)
        if i > 1 and i < len(beat_classifications) - 1:
            if (beat_classifications[i-2].get('label') == 'Normal' and
                beat_classifications[i-1].get('label') == 'Normal' and 
                beat.get('label') == 'PVC'):
                stats['trigeminy_episodes'] += 1
    
    return stats

def assess_clinical_urgency(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess clinical urgency and need for immediate intervention
    """
    urgency = {
        'level': 'routine',  # routine, urgent, emergent
        'reasons': [],
        'action_required': False
    }
    
    # Check for emergent conditions
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    
    # Extreme bradycardia
    if hr < 40 and hr > 0:
        urgency['level'] = 'emergent'
        urgency['reasons'].append('Severe bradycardia (<40 bpm)')
        urgency['action_required'] = True
    
    # Extreme tachycardia
    elif hr > 150:
        urgency['level'] = 'emergent'
        urgency['reasons'].append('Severe tachycardia (>150 bpm)')
        urgency['action_required'] = True
    
    # Very high PVC burden
    elif pvc_burden > 30:
        urgency['level'] = 'urgent'
        urgency['reasons'].append('Very high PVC burden (>30%)')
        urgency['action_required'] = True
    
    # Moderate abnormalities
    elif hr < 50 or hr > 120:
        urgency['level'] = 'urgent'
        urgency['reasons'].append('Abnormal heart rate')
    elif pvc_burden > 20:
        urgency['level'] = 'urgent'
        urgency['reasons'].append('High PVC burden')
    
    return urgency

def generate_critical_findings(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate critical findings card with color-coded severity levels
    
    Parameters:
    -----------
    analysis_results : dict
        ECG analysis results
    
    Returns:
    --------
    critical_findings : list
        List of critical findings with severity and color coding
    """
    findings = []
    
    hr = analysis_results.get('avg_heart_rate', 0)
    pvc_burden = analysis_results.get('pvc_burden', 0)
    sve_burden = analysis_results.get('sve_burden', 0)
    normal_pct = analysis_results.get('normal_percentage', 100)
    hrv_metrics = analysis_results.get('hrv_metrics', {})
    sdnn = hrv_metrics.get('sdnn', 50)
    rmssd = hrv_metrics.get('rmssd', 25)
    
    # Tachycardia detection
    if hr > 100:
        severity = 'high' if hr > 150 else 'moderate'
        findings.append({
            'title': 'Tachycardia',
            'value': f'{hr} bpm',
            'severity': severity,
            'description': f'Heart rate above normal range ({hr} bpm)',
            'icon': 'fa-heartbeat'
        })
    
    # Bradycardia detection
    if hr < 60 and hr > 0:
        severity = 'high' if hr < 40 else 'moderate'
        findings.append({
            'title': 'Bradycardia',
            'value': f'{hr} bpm',
            'severity': severity,
            'description': f'Heart rate below normal range ({hr} bpm)',
            'icon': 'fa-heartbeat'
        })
    
    # Low HRV detection
    if sdnn < 50:
        severity = 'high' if sdnn < 30 else 'moderate'
        findings.append({
            'title': 'Low HRV',
            'value': f'{sdnn:.1f} ms',
            'severity': severity,
            'description': f'Reduced heart rate variability (SDNN: {sdnn:.1f}ms)',
            'icon': 'fa-chart-line'
        })
    
    # PVC Burden > 5%
    if pvc_burden > 5:
        severity = 'critical' if pvc_burden > 20 else ('high' if pvc_burden > 10 else 'moderate')
        findings.append({
            'title': 'PVC Burden',
            'value': f'{pvc_burden:.1f}%',
            'severity': severity,
            'description': f'Premature ventricular contractions detected ({pvc_burden:.1f}%)',
            'icon': 'fa-exclamation-triangle'
        })
    
    # SVE Burden > 5%
    if sve_burden > 5:
        severity = 'high' if sve_burden > 10 else 'moderate'
        findings.append({
            'title': 'SVE Burden',
            'value': f'{sve_burden:.1f}%',
            'severity': severity,
            'description': f'Supraventricular ectopy detected ({sve_burden:.1f}%)',
            'icon': 'fa-exclamation-circle'
        })
    
    # Irregular rhythm
    if normal_pct < 85:
        severity = 'high' if normal_pct < 70 else 'moderate'
        findings.append({
            'title': 'Irregular Rhythm',
            'value': f'{normal_pct:.1f}% normal',
            'severity': severity,
            'description': f'Frequent arrhythmias detected ({100-normal_pct:.1f}% abnormal)',
            'icon': 'fa-wave-square'
        })
    
    # Sort findings by severity (critical > high > moderate > low)
    severity_order = {'critical': 0, 'high': 1, 'moderate': 2, 'low': 3}
    findings.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
    
    return findings