from flask import Flask, render_template, request, redirect, url_for
import threading
import time
import pandas as pd

app = Flask(__name__)

rooms = []
zones = []
utility_rates = {
    "electricity_price": None,
    "gas_price": None
}

@app.route("/")
def dashboard():
    return render_template("dashboard.html",rooms = rooms)

@app.route('/add_room', methods=['GET', 'POST'])
def add_room():
    if request.method == 'POST':
        name = request.form.get('roomName')
        room_size = request.form.get('roomSize')
        zone = request.form.get('zone')

        if name:
            from random import uniform, choice
            room = {
                'name': name,
                'room_size': int(room_size),
                'zone': zone,
                'temperature': round(uniform(65, 75), 1),
                'humidity': round(uniform(35, 55), 1),
                'occupancy': choice([True, False]),
                'heating_source': 'Electric Heater' if choice([True, False]) else 'Furnace'
            }
            if not room['occupancy']:
                room['heating_source'] = 'Off'

            rooms.append(room)
            return redirect(url_for('add_room'))

    return render_template('add_room.html', zones=zones)

@app.route('/config_home', methods=['GET', 'POST'])
def config_home():
    global utility_rates

    if request.method == 'POST':
        zone_name = request.form.get('zoneName')
        zone_desc = request.form.get('zoneDescription')

        if zone_name:
            zones.append({'name': zone_name, 'description': zone_desc})

        elec_price = request.form.get('electricityPrice')
        gas_price = request.form.get('gasPrice')

        if elec_price:
            utility_rates['electricity_price'] = float(elec_price)
        if gas_price:
            utility_rates['gas_price'] = float(gas_price)

        return redirect(url_for('config_home'))

    return render_template('config_home.html', zones=zones, rates=utility_rates)

@app.route('/demo')
def demo():
    global rooms, zones

    # Clear existing rooms and zones
    rooms.clear()
    zones.clear()
    
    file_path = "data-analytics/roommate_data/roommates_occupancy.csv"
    data = pd.read_csv(file_path)
    
    # Add zones
    unique_zones = data['zone'].unique()
    for zone in unique_zones:
        zones.append({'name': f'Zone {zone}', 'description': f'Zone {zone} description'})
    

    unique_rooms = data.groupby('room').first().reset_index()
    for _, row in unique_rooms.iterrows():
        room = {
            'name': row['room'],
            'zone': f'Zone {row["zone"]}',
            'temperature': 72,  # Default temperature
            'humidity': 44,     # Default humidity
            'occupancy': bool(row['occupied']),
            'heating_source': 'Furnace' if row['occupied'] else 'Off'
        }
        rooms.append(room)
    return redirect(url_for('dashboard'))

def update_heating_sources():
    while True:
        for room in rooms:
            if room['occupancy']:
                room['heating_source'] = 'Furnace'
            else:
                room['heating_source'] = 'Off'
        time.sleep(1)

threading.Thread(target=update_heating_sources, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)