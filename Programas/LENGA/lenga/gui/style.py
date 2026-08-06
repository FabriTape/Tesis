"""
Estilo visual oscuro (Dark Theme) profesional para PyQt5 en LENGA.
Paleta inspirada en Catppuccin Mocha / Material Dark.
"""

DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    font-size: 13px;
}

QWidget {
    color: #cdd6f4;
}

/* Headers y títulos */
QLabel {
    color: #cdd6f4;
}

QLabel#HeaderTitle {
    font-size: 20px;
    font-weight: bold;
    color: #89b4fa;
}

QLabel#SectionTitle {
    font-size: 15px;
    font-weight: bold;
    color: #b4befe;
    margin-top: 5px;
    margin-bottom: 5px;
}

QLabel#BadgeSuccess {
    background-color: #a6e3a1;
    color: #11111b;
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}

QLabel#BadgeWarning {
    background-color: #f38ba8;
    color: #11111b;
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}

/* Botones */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton#PrimaryButton {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
}

QPushButton#PrimaryButton:hover {
    background-color: #b4befe;
}

QPushButton#SuccessButton {
    background-color: #a6e3a1;
    color: #11111b;
    border: none;
}

QPushButton#SuccessButton:hover {
    background-color: #94e2d5;
}

QPushButton#DangerButton {
    background-color: #f38ba8;
    color: #11111b;
    border: none;
}

/* Entradas de texto y números */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 6px 10px;
    selection-background-color: #89b4fa;
    selection-color: #11111b;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #89b4fa;
}

/* Pestañas */
QTabWidget::pane {
    border: 1px solid #313244;
    background-color: #181825;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #313244;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #262738;
    color: #cdd6f4;
}

/* Tablas */
QTableWidget {
    background-color: #181825;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 6px;
}

QHeaderView::section {
    background-color: #313244;
    color: #cdd6f4;
    padding: 6px;
    font-weight: bold;
    border: none;
}

QTableWidget::item:selected {
    background-color: #45475a;
    color: #89b4fa;
}

/* GroupBox y Tarjetas */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #b4befe;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

/* CheckBox y RadioButton */
QCheckBox, QRadioButton {
    spacing: 8px;
    color: #cdd6f4;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #585b70;
    background-color: #181825;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

/* ScrollBar */
QScrollBar:vertical {
    border: none;
    background-color: #181825;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}
"""
