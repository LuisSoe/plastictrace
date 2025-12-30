"""
MapView: QWebEngine + Leaflet mapping widget with QWebChannel bridge.
"""
import os
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QFrame, QScrollArea
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl, QTimer, QSettings, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import Qt

# Try to import location services
try:
    from PyQt5.QtPositioning import QGeoPositionInfoSource, QGeoPositionInfo, QGeoCoordinate
    LOCATION_AVAILABLE = True
except ImportError:
    LOCATION_AVAILABLE = False
    print("Warning: QtLocation not available. GPS features will be disabled.")
from domain.models import Location
from domain.geo import filter_locations, haversine_distance
from ml.config import CLASSES


class MapBridge(QObject):
    """Bridge object for QWebChannel communication."""
    
    markerClicked = pyqtSignal(str)  # location id
    userLocationFromBrowser = pyqtSignal(float, float)  # lat, lon
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @pyqtSlot(float, float)
    def navigate(self, lat: float, lon: float):
        """Open Google Maps navigation."""
        url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
        QDesktopServices.openUrl(QUrl(url))
    
    @pyqtSlot(str)
    def openUrl(self, url: str):
        """Open URL in browser."""
        QDesktopServices.openUrl(QUrl(url))
    
    @pyqtSlot(str)
    def onMarkerClicked(self, location_id: str):
        """Handle marker click."""
        self.markerClicked.emit(location_id)
    
    @pyqtSlot(float, float)
    def setUserLocationFromBrowser(self, lat: float, lon: float):
        """Set user location from browser geolocation."""
        self.userLocationFromBrowser.emit(lat, lon)


class MaterialChip(QPushButton):
    """Material design chip for plastic type selection."""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: 2px solid #475569;
                border-radius: 16px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #10b981;
                border-color: #10b981;
                color: white;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:checked:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border-color: #334155;
            }
        """)


class LocationCard(QFrame):
    """Card widget for location list item."""
    
    navigateClicked = pyqtSignal(str)  # location id
    
    def __init__(self, location: Location, parent=None):
        super().__init__(parent)
        self.location = location
        self.setup_ui()
    
    def setup_ui(self):
        """Setup card UI."""
        self.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame:hover {
                background-color: #334155;
                border-color: #10b981;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Name
        name_label = QLabel(self.location.name)
        name_label.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold;")
        layout.addWidget(name_label)
        
        # Address
        address_label = QLabel(self.location.address)
        address_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        address_label.setWordWrap(True)
        layout.addWidget(address_label)
        
        # Distance and types
        info_layout = QHBoxLayout()
        
        distance_label = QLabel(f"{self.location.distance_km:.2f} km")
        distance_label.setStyleSheet("color: #10b981; font-size: 12px; font-weight: bold;")
        info_layout.addWidget(distance_label)
        
        info_layout.addStretch()
        
        types_label = QLabel(", ".join(self.location.types[:3]))
        types_label.setStyleSheet("color: #64748b; font-size: 11px;")
        info_layout.addWidget(types_label)
        
        layout.addLayout(info_layout)
        
        # Hours and contact
        if self.location.hours:
            hours_label = QLabel(f"⏰ {self.location.hours}")
            hours_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
            layout.addWidget(hours_label)
        
        if self.location.phone:
            phone_label = QLabel(f"📞 {self.location.phone}")
            phone_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
            layout.addWidget(phone_label)
        
        # Navigate button
        nav_btn = QPushButton("Navigasi")
        nav_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                margin-top: 8px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        nav_btn.clicked.connect(lambda: self.navigateClicked.emit(self.location.id))
        layout.addWidget(nav_btn)


class MapView(QWidget):
    """Map view widget with search, filters, and location list."""
    
    locationSelected = pyqtSignal(str)  # location id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Settings
        self.settings = QSettings("PlastiTrace", "MapView")
        
        # State
        self.user_lat = -6.2297  # Jakarta Selatan
        self.user_lon = 106.7997
        self.locations: list[Location] = []
        self.filtered_locations: list[Location] = []
        self.selected_types: set[str] = set()
        self.auto_follow = False  # Auto-follow user location
        
        # Map bridge
        self.map_bridge = MapBridge(self)
        self.map_bridge.markerClicked.connect(self.on_marker_clicked)
        self.map_bridge.userLocationFromBrowser.connect(self.set_user_location_from_browser)
        
        # Debounce timer
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._apply_search)
        
        # GPS location source
        self.position_source = None
        if LOCATION_AVAILABLE:
            try:
                # Check if any GPS sources are available before creating one
                # This avoids the "No known GPS device found" warning
                available_sources = QGeoPositionInfoSource.availableSources()
                if available_sources:
                    self.position_source = QGeoPositionInfoSource.createDefaultSource(self)
                    if self.position_source:
                        self.position_source.positionUpdated.connect(self._on_position_updated)
                        self.position_source.error.connect(self._on_position_error)
                        # Update every 5 seconds when active
                        self.position_source.setUpdateInterval(5000)
                # If no GPS sources available, position_source remains None
                # Browser geolocation will be used as fallback
            except Exception as e:
                # Silently fail - browser geolocation will be used as fallback
                self.position_source = None
        
        self.setup_ui()
        self._load_settings()
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Top: Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Cari lokasi...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #10b981;
            }
        """)
        self.search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.search_input)
        
        # Use current location button
        self.use_location_btn = QPushButton("📍")
        self.use_location_btn.setToolTip("Gunakan lokasi saat ini")
        self.use_location_btn.setFixedSize(40, 40)
        self.use_location_btn.setCheckable(True)
        self.use_location_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #475569;
            }
            QPushButton:checked {
                background-color: #10b981;
                border-color: #10b981;
            }
        """)
        self.use_location_btn.clicked.connect(self._toggle_location_follow)
        search_layout.addWidget(self.use_location_btn)
        
        layout.addLayout(search_layout)
        
        # Radius slider - covers entire Indonesia (Jakarta to Ambon ~2500km)
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Max Distance:"))
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(4000)  # Covers entire Indonesia (Jakarta to Merauke ~3700km, Ambon ~2500km)
        # Default to 4000km to cover all Indonesia including easternmost points
        default_radius = int(self.settings.value("radius_km", 4000))
        self.radius_slider.setValue(default_radius)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        radius_layout.addWidget(self.radius_slider)
        
        self.radius_label = QLabel(f"{self.radius_slider.value()} km")
        self.radius_label.setStyleSheet("color: #f8fafc; font-weight: bold; min-width: 80px;")
        radius_layout.addWidget(self.radius_label)
        
        info_label = QLabel("(Showing all locations)")
        info_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        radius_layout.addWidget(info_label)
        
        layout.addLayout(radius_layout)
        
        # Material type chips
        chips_layout = QHBoxLayout()
        chips_layout.addWidget(QLabel("Filter:"))
        self.chips = {}
        for plastic_type in CLASSES:
            chip = MaterialChip(plastic_type)
            chip.toggled.connect(self._on_chip_toggled)
            self.chips[plastic_type] = chip
            chips_layout.addWidget(chip)
        chips_layout.addStretch()
        layout.addLayout(chips_layout)
        
        # Map widget
        self.map_web = QWebEngineView()
        self._init_map()
        layout.addWidget(self.map_web, 2)
        
        # Location list
        list_label = QLabel("Lokasi Terdekat")
        list_label.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold;")
        layout.addWidget(list_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
            }
        """)
        
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setSpacing(8)
        scroll.setWidget(self.list_container)
        
        layout.addWidget(scroll, 1)
    
    def _init_map(self):
        """Initialize Leaflet map."""
        html_path = Path(__file__).parent.parent / "assets" / "map" / "leaflet.html"
        url = QUrl.fromLocalFile(str(html_path.absolute()))
        self.map_web.load(url)
        
        # Setup QWebChannel
        channel = QWebChannel(self.map_web.page())
        channel.registerObject("bridge", self.map_bridge)
        self.map_web.page().setWebChannel(channel)
        
        # Wait for page load
        self.map_web.page().loadFinished.connect(self._on_map_loaded)
    
    def _on_map_loaded(self, success: bool):
        """Handle map page load."""
        if success:
            # Set user location (wait a bit for bridge to initialize)
            js_code = f"""
            setTimeout(function() {{
                if (typeof window.bridge !== 'undefined' && window.bridge && window.bridge.setUserLocation) {{
                    window.bridge.setUserLocation({self.user_lat}, {self.user_lon});
                }}
            }}, 100);
            """
            self.map_web.page().runJavaScript(js_code)
            
            # Update map with locations if they were set before map loaded
            if self.locations:
                # Wait a bit more for bridge to be fully ready
                QTimer.singleShot(200, self._update_map)
    
    def set_user_location(self, lat: float, lon: float):
        """Set user's current location."""
        self.user_lat = lat
        self.user_lon = lon
        self.settings.setValue("user_lat", lat)
        self.settings.setValue("user_lon", lon)
        
        if self.map_web.page():
            js_code = f"""
            if (typeof window.bridge !== 'undefined' && window.bridge && window.bridge.setUserLocation) {{
                window.bridge.setUserLocation({lat}, {lon});
            }}
            """
            self.map_web.page().runJavaScript(js_code)
        self._update_map()
    
    def set_user_location_from_browser(self, lat: float, lon: float):
        """Set user location from browser geolocation (called via signal)."""
        self.set_user_location(lat, lon)
        # If auto-follow is enabled, center map on user location
        if self.auto_follow and self.map_web.page():
            js_code = f"""
            if (typeof window.bridge !== 'undefined' && window.bridge && window.bridge.centerOnUser) {{
                window.bridge.centerOnUser({lat}, {lon});
            }}
            """
            self.map_web.page().runJavaScript(js_code)
    
    def set_locations(self, locations: list[Location]):
        """Set locations to display."""
        self.locations = locations
        print(f"MapView: Set {len(locations)} locations")
        self._update_map()
    
    def set_selected_plastic_type(self, plastic_type: str):
        """Set selected plastic type and filter locations."""
        if plastic_type == "Unknown":
            self.selected_types.clear()
        else:
            self.selected_types = {plastic_type}
        
        # Update chips
        for chip_type, chip in self.chips.items():
            chip.setChecked(chip_type in self.selected_types)
            chip.setEnabled(plastic_type != "Unknown")
        
        self._update_map()
    
    def _update_map(self):
        """Update map with filtered locations."""
        # Filter locations
        types_list = list(self.selected_types) if self.selected_types else []
        radius = self.radius_slider.value()
        
        # Show ALL locations within radius (no limit)
        # With radius set to 4000km (covers all Indonesia), this will show all locations from anywhere in Indonesia
        self.filtered_locations = filter_locations(
            self.locations,
            self.user_lat,
            self.user_lon,
            radius_km=radius,  # Use the radius value (default 4000km covers all Indonesia)
            types_selected=types_list,
            max_results=None  # Show all locations, no limit
        )
        
        if self.filtered_locations:
            max_dist = max(loc.distance_km or 0 for loc in self.filtered_locations)
            print(f"MapView: Showing {len(self.filtered_locations)} locations (max distance: {max_dist:.1f}km, types={types_list})")
        else:
            print(f"MapView: No locations found (types={types_list})")
        
        if not self.map_web.page():
            # Map not ready yet, will be called again when map loads
            print("MapView: Map page not ready, will update when loaded")
            return
        
        # Build locations data as JSON (safer than string concatenation)
        import json
        locations_data = []
        for loc in self.filtered_locations:
            locations_data.append({
                'id': loc.id,
                'name': loc.name,
                'lat': loc.lat,
                'lon': loc.lon,
                'address': loc.address,
                'distance': loc.distance_km or 0,
                'types': loc.types
            })
        
        # Convert to JSON string (properly escaped)
        locations_json = json.dumps(locations_data)
        
        # Single JavaScript call to add all markers
        js_code = f"""
        (function() {{
            if (typeof window.bridge === 'undefined' || !window.bridge) {{
                console.log('Bridge not ready, retrying in 100ms...');
                setTimeout(arguments.callee, 100);
                return;
            }}
            
            // Clear existing markers
            if (window.bridge.clearMarkers) {{
                window.bridge.clearMarkers();
            }}
            
            // Parse locations from JSON
            var locations = {locations_json};
            
            // Add all locations
            console.log('Starting to add', locations.length, 'locations to map');
            var addedCount = 0;
            for (var i = 0; i < locations.length; i++) {{
                var loc = locations[i];
                try {{
                    if (window.bridge.addLocation) {{
                        window.bridge.addLocation(loc.id, loc.name, loc.lat, loc.lon, loc.address, loc.distance, loc.types);
                        addedCount++;
                    }} else {{
                        console.error('window.bridge.addLocation is not available');
                    }}
                }} catch(e) {{
                    console.error('Error adding location:', e, loc);
                }}
            }}
            
            console.log('Successfully added', addedCount, 'out of', locations.length, 'locations');
            
            // Fit bounds if there are locations
            if (locations.length > 0 && window.bridge.fitBounds) {{
                setTimeout(function() {{
                    window.bridge.fitBounds();
                    console.log('Fitted bounds to show all markers');
                }}, 100);
            }}
        }})();
        """
        self.map_web.page().runJavaScript(js_code)
        
        # Update list
        self._update_list()
    
    def _update_list(self):
        """Update location list."""
        # Clear existing cards
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.filtered_locations:
            # Empty state
            empty_label = QLabel("Tidak ada lokasi ditemukan.")
            empty_label.setStyleSheet("color: #64748b; font-size: 12px; padding: 20px;")
            empty_label.setAlignment(Qt.AlignCenter)
            self.list_layout.addWidget(empty_label)
            return
        
        # Add location cards
        for loc in self.filtered_locations:
            card = LocationCard(loc)
            card.navigateClicked.connect(self._on_navigate_clicked)
            self.list_layout.addWidget(card)
        
        self.list_layout.addStretch()
    
    def _on_search_changed(self, text: str):
        """Handle search text change (debounced)."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms debounce
    
    def _apply_search(self):
        """Apply search filter."""
        # TODO: Implement search filtering
        pass
    
    def _on_radius_changed(self, value: int):
        """Handle radius slider change."""
        self.radius_label.setText(f"{value} km")
        self.settings.setValue("radius_km", value)
        self._update_map()
    
    def _on_chip_toggled(self, checked: bool):
        """Handle material chip toggle."""
        chip = self.sender()
        chip_type = None
        for t, c in self.chips.items():
            if c == chip:
                chip_type = t
                break
        
        if chip_type:
            if checked:
                self.selected_types.add(chip_type)
            else:
                self.selected_types.discard(chip_type)
            self._update_map()
    
    def _toggle_location_follow(self, checked: bool):
        """Toggle auto-follow location mode."""
        self.auto_follow = checked
        
        if checked:
            # Start location updates
            if self.position_source:
                try:
                    self.position_source.startUpdates()
                    self.use_location_btn.setToolTip("Mengikuti lokasi Anda (klik untuk berhenti)")
                except Exception as e:
                    print(f"Error starting location updates: {e}")
                    self.use_location_btn.setChecked(False)
                    self.auto_follow = False
            else:
                # Fallback: use browser geolocation with watch
                self._request_browser_location(watch=True)
                self.use_location_btn.setToolTip("Mengikuti lokasi Anda (klik untuk berhenti)")
        else:
            # Stop location updates
            if self.position_source:
                self.position_source.stopUpdates()
            else:
                # Stop browser location watch
                self._stop_browser_location_watch()
            self.use_location_btn.setToolTip("Gunakan lokasi saat ini")
    
    def _request_browser_location(self, watch=False):
        """Request location from browser (fallback if Qt location not available)."""
        if self.map_web.page():
            if watch:
                # Watch position for continuous updates
                js_code = """
                if (navigator.geolocation) {
                    if (window.watchId) {
                        navigator.geolocation.clearWatch(window.watchId);
                    }
                    window.watchId = navigator.geolocation.watchPosition(
                        function(position) {
                            if (window.bridge && window.bridge.setUserLocationFromBrowser) {
                                window.bridge.setUserLocationFromBrowser(position.coords.latitude, position.coords.longitude);
                            }
                            if (window.bridge && window.bridge.centerOnUser) {
                                window.bridge.centerOnUser(position.coords.latitude, position.coords.longitude);
                            }
                        },
                        function(error) {
                            console.error('Geolocation error:', error);
                            if (error.code === 1) {
                                alert('Akses lokasi ditolak. Pastikan izin lokasi diberikan di pengaturan browser.');
                            }
                        },
                        {
                            enableHighAccuracy: true,
                            timeout: 10000,
                            maximumAge: 5000
                        }
                    );
                } else {
                    alert('Browser tidak mendukung geolocation.');
                }
                """
            else:
                # One-time position request
                js_code = """
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        function(position) {
                            if (window.bridge && window.bridge.setUserLocationFromBrowser) {
                                window.bridge.setUserLocationFromBrowser(position.coords.latitude, position.coords.longitude);
                            }
                        },
                        function(error) {
                            console.error('Geolocation error:', error);
                            alert('Tidak dapat mengakses lokasi. Pastikan izin lokasi diberikan.');
                        }
                    );
                } else {
                    alert('Browser tidak mendukung geolocation.');
                }
                """
            self.map_web.page().runJavaScript(js_code)
    
    def _stop_browser_location_watch(self):
        """Stop watching browser location."""
        if self.map_web.page():
            js_code = """
            if (window.watchId) {
                navigator.geolocation.clearWatch(window.watchId);
                window.watchId = null;
            }
            """
            self.map_web.page().runJavaScript(js_code)
    
    def _on_position_updated(self, position: 'QGeoPositionInfo'):
        """Handle GPS position update."""
        if position.isValid():
            coord = position.coordinate()
            if coord.isValid():
                lat = coord.latitude()
                lon = coord.longitude()
                self.set_user_location(lat, lon)
                
                # Center map on user location if auto-follow is enabled
                if self.auto_follow and self.map_web.page():
                    js_code = f"""
                    if (typeof window.bridge !== 'undefined' && window.bridge && window.bridge.centerOnUser) {{
                        window.bridge.centerOnUser({lat}, {lon});
                    }}
                    """
                    self.map_web.page().runJavaScript(js_code)
    
    def _on_position_error(self, error: int):
        """Handle GPS position error."""
        error_messages = {
            0: "Tidak ada error",
            1: "Akses ditolak",
            2: "Tidak ada posisi tersedia",
            3: "Timeout"
        }
        msg = error_messages.get(error, f"Error {error}")
        print(f"GPS Error: {msg}")
        self.use_location_btn.setChecked(False)
        self.auto_follow = False
    
    def on_marker_clicked(self, location_id: str):
        """Handle marker click."""
        # Find location and scroll to it in list
        for i, loc in enumerate(self.filtered_locations):
            if loc.id == location_id:
                # TODO: Scroll to card
                self.locationSelected.emit(location_id)
                break
    
    def _on_navigate_clicked(self, location_id: str):
        """Handle navigate button click."""
        for loc in self.filtered_locations:
            if loc.id == location_id:
                url = f"https://www.google.com/maps/dir/?api=1&destination={loc.lat},{loc.lon}"
                QDesktopServices.openUrl(QUrl(url))
                break
    
    def _load_settings(self):
        """Load saved settings."""
        self.user_lat = float(self.settings.value("user_lat", -6.2297))
        self.user_lon = float(self.settings.value("user_lon", 106.7997))
        radius = int(self.settings.value("radius_km", 4000))  # Default covers all Indonesia
        self.radius_slider.setValue(radius)
        self.radius_label.setText(f"{radius} km")

