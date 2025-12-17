"""
Map widget for displaying nearby waste disposal/recycling locations.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint
from ml.config import CLASSES


# Sample location data (in real app, this could come from a database or API)
# Format: {category: [(name, address, lat, lon, distance_km), ...]}
SAMPLE_LOCATIONS = {
    "HDPE": [
        ("Bank Sampah Jakarta Selatan", "Jl. Kebayoran Baru, Jakarta Selatan", -6.2297, 106.7997, 2.5),
        ("TPS 3R Cilandak", "Jl. Cilandak KKO, Jakarta Selatan", -6.2892, 106.8000, 3.2),
        ("Bank Sampah Kemang", "Jl. Kemang Raya, Jakarta Selatan", -6.2600, 106.8100, 4.1),
    ],
    "PET": [
        ("Bank Sampah Jakarta Selatan", "Jl. Kebayoran Baru, Jakarta Selatan", -6.2297, 106.7997, 2.5),
        ("TPS 3R Cilandak", "Jl. Cilandak KKO, Jakarta Selatan", -6.2892, 106.8000, 3.2),
        ("Recycling Center Senayan", "Jl. Asia Afrika, Jakarta Pusat", -6.2275, 106.8000, 5.0),
    ],
    "PP": [
        ("Bank Sampah Jakarta Selatan", "Jl. Kebayoran Baru, Jakarta Selatan", -6.2297, 106.7997, 2.5),
        ("TPS 3R Cilandak", "Jl. Cilandak KKO, Jakarta Selatan", -6.2892, 106.8000, 3.2),
        ("Waste Management Center", "Jl. Sudirman, Jakarta Pusat", -6.2088, 106.8000, 6.5),
    ],
    "PS": [
        ("TPS Residu Jakarta Selatan", "Jl. Fatmawati, Jakarta Selatan", -6.2800, 106.7900, 3.8),
        ("Landfill Bantar Gebang", "Bekasi, Jawa Barat", -6.3000, 107.0000, 25.0),
        ("Waste Processing Facility", "Jl. Ciputat Raya, Jakarta Selatan", -6.3100, 106.7500, 7.2),
    ],
}

# Default location (user's location - could be from GPS)
DEFAULT_LAT = -6.2297  # Jakarta Selatan
DEFAULT_LON = 106.7997


class MapWidget(QWidget):
    """Widget for displaying a simple map with location markers."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.locations = []  # List of (name, lat, lon, distance_km)
        self.user_lat = DEFAULT_LAT
        self.user_lon = DEFAULT_LON
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1e293b; border-radius: 10px;")
    
    def set_locations(self, locations):
        """
        Set locations to display.
        
        Args:
            locations: List of (name, address, lat, lon, distance_km) tuples
        """
        self.locations = locations
        self.update()  # Trigger repaint
    
    def set_user_location(self, lat, lon):
        """Set user's current location."""
        self.user_lat = lat
        self.user_lon = lon
        self.update()
    
    def paintEvent(self, event):
        """Draw the map with markers."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        widget_width = self.width()
        widget_height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, widget_width, widget_height, QColor(30, 41, 59))
        
        if not self.locations:
            # Draw placeholder text
            painter.setPen(QColor(148, 163, 184))
            font = QFont("Arial", 12)
            painter.setFont(font)
            painter.drawText(
                QRect(0, 0, widget_width, widget_height),
                Qt.AlignCenter,
                "Pilih kategori plastik\ndan deteksi objek\nuntuk melihat lokasi terdekat"
            )
            return
        
        # Calculate bounds
        lats = [loc[2] for loc in self.locations] + [self.user_lat]
        lons = [loc[3] for loc in self.locations] + [self.user_lon]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Add padding
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        if lat_range < 0.01:
            lat_range = 0.01
        if lon_range < 0.01:
            lon_range = 0.01
        
        min_lat -= lat_range * 0.1
        max_lat += lat_range * 0.1
        min_lon -= lon_range * 0.1
        max_lon += lon_range * 0.1
        
        # Map function: convert lat/lon to screen coordinates
        def map_coords(lat, lon):
            x = int(((lon - min_lon) / (max_lon - min_lon)) * (widget_width - 40) + 20)
            y = int(((max_lat - lat) / (max_lat - min_lat)) * (widget_height - 40) + 20)
            return (x, y)
        
        # Draw grid lines (optional)
        painter.setPen(QPen(QColor(51, 65, 85), 1, Qt.DashLine))
        for i in range(3):
            x = int(widget_width * (i + 1) / 4)
            painter.drawLine(x, 0, x, widget_height)
        for i in range(3):
            y = int(widget_height * (i + 1) / 4)
            painter.drawLine(0, y, widget_width, y)
        
        # Draw user location (blue marker)
        user_x, user_y = map_coords(self.user_lat, self.user_lon)
        painter.setBrush(QBrush(QColor(59, 130, 246)))  # Blue
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(user_x - 8, user_y - 8, 16, 16)
        painter.setPen(QPen(QColor(59, 130, 246), 1))
        painter.drawText(user_x + 12, user_y - 8, "Anda")
        
        # Draw location markers
        colors = [
            QColor(16, 185, 129),  # Green
            QColor(245, 158, 11),  # Yellow
            QColor(239, 68, 68),   # Red
        ]
        
        for i, (name, address, lat, lon, distance) in enumerate(self.locations):
            x, y = map_coords(lat, lon)
            
            # Choose color based on distance
            if distance < 3:
                color = colors[0]  # Green (close)
            elif distance < 5:
                color = colors[1]  # Yellow (medium)
            else:
                color = colors[2]  # Red (far)
            
            # Draw marker
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(x - 6, y - 6, 12, 12)
            
            # Draw distance label
            painter.setPen(QPen(color, 1))
            font = QFont("Arial", 9, QFont.Bold)
            painter.setFont(font)
            label = f"{i+1}. {distance:.1f} km"
            painter.drawText(x + 10, y + 5, label)
        
        # Draw legend
        legend_y = widget_height - 80
        painter.setPen(QColor(248, 250, 252))
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.drawText(10, legend_y, "Legenda:")
        
        legend_items = [
            (colors[0], "< 3 km"),
            (colors[1], "3-5 km"),
            (colors[2], "> 5 km"),
        ]
        
        for i, (color, text) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 20
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(10, y_pos - 6, 12, 12)
            painter.setPen(QColor(248, 250, 252))
            painter.drawText(30, y_pos + 5, text)


class LocationListWidget(QWidget):
    """Widget for displaying list of locations with details."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("Lokasi Terdekat")
        title.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 5px;
                color: #f8fafc;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #334155;
            }
            QListWidget::item:selected {
                background-color: #1e293b;
            }
        """)
        layout.addWidget(self.list_widget)
    
    def set_locations(self, locations):
        """
        Set locations to display.
        
        Args:
            locations: List of (name, address, lat, lon, distance_km) tuples
        """
        self.list_widget.clear()
        
        for i, (name, address, lat, lon, distance) in enumerate(locations):
            item_text = f"{i+1}. {name}\n   {address}\n   Jarak: {distance:.1f} km"
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

