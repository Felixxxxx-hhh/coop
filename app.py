
import numpy as np
import pandas as pd
import joblib
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
models = joblib.load('models.pkl')
model2 = joblib.load('model2.pkl')

# Load dropdown data
with open('dropdown_data.json', 'r', encoding='utf-8') as f:
    dropdown_data = json.load(f)

# 直接这样写就行，不需要先赋值再覆盖
VENDOR_LIST = sorted(dropdown_data['vendors'])
SHIP_FROM_LIST = sorted(dropdown_data['ship_froms'])
SHIP_VIA_LIST = sorted(dropdown_data['ship_vias'])
ITEM_LIST = sorted(dropdown_data['items'])
INCOTERM_LIST = sorted(dropdown_data['incoterms'])

# ========== 规则 3: Incoterm 费用强制归零规则 ==========
INCOTERM_RULES = {
    'EXW': ['Freight(O)', 'Local(Q)', 'Brokerage(S)'],
    'FCA': ['Freight(O)', 'Local(Q)', 'Brokerage(S)'],
    'FOB': ['Freight(O)', 'Local(Q)', 'Brokerage(S)'],
    'CFR': ['Local(Q)', 'Brokerage(S)'],
    'CIF': ['Local(Q)', 'Brokerage(S)'],
    'CPT': ['Local(Q)', 'Brokerage(S)'],
    'DAP': ['Brokerage(S)'],
    'DDP': [],
}

# ========== 规则 1: BY AIR 发货地限制 ==========
AIR_SHIP_FROMS = [
    'HK BY AIR', 'TAIWAN KEELUNG BY AIR', 'JAPAN OSAKA BY AIR',
    'VIETNAM BY AIR', 'CHINA SHANGHAI BY AIR', 'JAPAN OSAKA TO MM BY AIR',
    'ITALY BY AIR', 'KOREA BUSAN BY AIR', 'TAIWAN KEELUNG TO MM BY AIR',
    'HK TO MM BY AIR', 'FRANCE BY AIR', 'FRANCE TO MM BY AIR',
    'CHINA GUANGZHOU BY AIR', 'KOREA BUSAN TO MM BY AIR', 'CHINA XIAMEN BY AIR',
    'CHINA SHANGHAI TO MM BY AIR', 'VIETNAM TO MM BY AIR', 'NEW ZELAND BY AIR',
    'TAIWAN KAOHSIUNG TO MM BY AIR', 'THAILAND TO MM BY AIR',
    'TAIWAN KAOHSIUNG BY AIR', 'CHINA SHENZHEN BY AIR'
]

# ========== 规则 2: Incoterm + 快递方式限制 ==========
INVALID_INCOTERM_VIA = [
    ('FOB', 'DHL'), ('FOB', 'FED'),
    ('CIF', 'FED'),
    ('DAP', 'DHL'), ('DAP', 'FED'),
]

def is_valid_combination(ship_from, ship_via, incoterm):
    ship_via_upper = ship_via.upper()
    if ship_from in AIR_SHIP_FROMS:
        if not any(k in ship_via_upper for k in ['AIR', 'DHL', 'FED']):
            return False, f"'{ship_from}' can only use AIR/DHL/FED, not '{ship_via}'"
    for inv_inc, inv_via in INVALID_INCOTERM_VIA:
        if incoterm == inv_inc and inv_via in ship_via_upper:
            return False, f"'{incoterm}' cannot be used with '{ship_via}'"
    return True, None

def predict_cost(vendor_name, ship_from, ship_via, item, total_material_cost, unit_price, incoterm):
    ship_from_via = f"{ship_from}_{ship_via}"
    vendor_from_via = f"{vendor_name}_{ship_from_via}"
    vendor_item = f"{vendor_name}_{item}"
    
    zero_cols = INCOTERM_RULES.get(incoterm, [])
    exwork_is_zero = 1 if 'Exwork(M)' in zero_cols else 0
    freight_is_zero = 1 if 'Freight(O)' in zero_cols else 0
    local_is_zero = 1 if 'Local(Q)' in zero_cols else 0
    brokerage_is_zero = 1 if 'Brokerage(S)' in zero_cols else 0
    
    input_data = pd.DataFrame({
        'Vendor_From_Via': [vendor_from_via],
        'Incoterm': [incoterm],
        'Item': [item],
        'Total_Material_Cost': [np.log1p(max(total_material_cost, 0.01))],
        'Unit_Price': [np.log1p(max(unit_price, 0.01))],
        'Exwork_is_zero': [exwork_is_zero],
        'Freight_is_zero': [freight_is_zero],
        'Local_is_zero': [local_is_zero],
        'Brokerage_is_zero': [brokerage_is_zero],
    })
    
    for col in ['Vendor_From_Via', 'Incoterm', 'Item']:
        input_data[col] = input_data[col].astype('category')
    
    results = {}
    for target in ['Exwork(M)', 'Freight(O)', 'Local(Q)', 'Brokerage(S)']:
        pred = models[target].predict(input_data)[0]
        results[target] = max(0, float(np.expm1(pred)))
    
    # 规则 4: DHL/FED 时 Exwork 强制为 0
    if any(k in ship_via.upper() for k in ['DHL', 'FED']):
        results['Exwork(M)'] = 0.0
    
    # 规则 3: 根据 Incoterm 强制归零
    for col in zero_cols:
        results[col] = 0.0
    
    results['Total_Import_cost(U)'] = sum(results.values())
    
    return results

@app.route('/')
def index():
    return render_template('index.html', 
                         vendors=VENDOR_LIST,
                         ship_froms=SHIP_FROM_LIST,
                         ship_vias=SHIP_VIA_LIST,
                         incoterms=INCOTERM_LIST,
                         items=ITEM_LIST)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        vendor = data.get('vendor')
        ship_from = data.get('ship_from')
        ship_via = data.get('ship_via')
        item = data.get('item')
        material_cost = float(data.get('material_cost', 50000))
        unit_price = float(data.get('unit_price', 100))
        incoterm = data.get('incoterm')
        
        valid, reason = is_valid_combination(ship_from, ship_via, incoterm)
        if not valid:
            return jsonify({'error': reason}), 400
        
        result = predict_cost(vendor, ship_from, ship_via, item, material_cost, unit_price, incoterm)
        
        return jsonify({
            'success': True,
            'exwork': round(result['Exwork(M)'], 2),
            'freight': round(result['Freight(O)'], 2),
            'local': round(result['Local(Q)'], 2),
            'brokerage': round(result['Brokerage(S)'], 2),
            'total': round(result['Total_Import_cost(U)'], 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        
        item = data.get('item')
        material_cost = float(data.get('material_cost', 50000))
        unit_price = float(data.get('unit_price', 100))
        
        vendors = data.get('vendors', [])
        ship_froms = data.get('ship_froms', [])
        ship_vias = data.get('ship_vias', [])
        incoterms = data.get('incoterms', [])
        top_n = data.get('top_n', 5)
        
        if not vendors:
            vendors = VENDOR_LIST
        if not ship_froms:
            ship_froms = SHIP_FROM_LIST
        if not ship_vias:
            ship_vias = SHIP_VIA_LIST
        if not incoterms:
            incoterms = INCOTERM_LIST
        
        print(f"Optimizing: item={item}, material_cost={material_cost}, unit_price={unit_price}")
        
        results = []
        for vendor in vendors:
            for ship_from in ship_froms:
                for ship_via in ship_vias:
                    for incoterm in incoterms:
                        valid, _ = is_valid_combination(ship_from, ship_via, incoterm)
                        if not valid:
                            continue
                        
                        try:
                            result = predict_cost(vendor, ship_from, ship_via, item, material_cost, unit_price, incoterm)
                            results.append({
                                'vendor': vendor,
                                'ship_from': ship_from,
                                'ship_via': ship_via,
                                'incoterm': incoterm,
                                'exwork': round(result['Exwork(M)'], 2),
                                'freight': round(result['Freight(O)'], 2),
                                'local': round(result['Local(Q)'], 2),
                                'brokerage': round(result['Brokerage(S)'], 2),
                                'total': round(result['Total_Import_cost(U)'], 2)
                            })
                        except Exception:
                            continue
        
        results.sort(key=lambda x: x['total'])
        top_results = results[:top_n]
        
        print(f"Found {len(results)} valid combinations")
        
        return jsonify({
            'success': True,
            'combinations': top_results,
            'total_searched': len(results)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
