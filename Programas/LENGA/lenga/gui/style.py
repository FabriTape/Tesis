"""
Estilo visual "Otoño Austral" para PyQt5 en LENGA.
Inspirado en la transición de la lenga patagónica (rojo/anaranjado fuego y verde musgo),
la corteza grisácea del árbol y la profundidad por capas esmeriladas estilo macOS Sequoia.
"""

DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1E2022;
    color: #ECEFF1;
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Ubuntu, sans-serif;
    font-size: 13px;
}

QWidget {
    color: #ECEFF1;
}

/* Contenedor del título superior */
QFrame#HeaderTitleBox {
    background-color: rgba(0, 0, 0, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
}

/* Headers y títulos */
QLabel {
    color: #ECEFF1;
}

QLabel#HeaderTitle {
    font-size: 22px;
    font-weight: bold;
    color: #D95C14;
    letter-spacing: 1px;
}

QLabel#SectionTitle {
    font-size: 15px;
    font-weight: bold;
    color: #F0A060;
    margin-top: 6px;
    margin-bottom: 6px;
}

QLabel#BadgeSuccess {
    background-color: #4A7A3A;
    color: #FFFFFF;
    font-weight: bold;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #5C9449;
}

QLabel#BadgeWarning {
    background-color: #D94E4E;
    color: #FFFFFF;
    font-weight: bold;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #EE6060;
}

/* Botones con micro-bisel superior y esquinas suavizadas */
QPushButton {
    background-color: rgba(44, 46, 48, 0.85);
    color: #ECEFF1;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: rgba(58, 61, 64, 0.95);
    border-color: #D95C14;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: rgba(34, 36, 38, 0.95);
}

QPushButton#PrimaryButton {
    background-color: #D95C14;
    color: #FFFFFF;
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.28);
    font-weight: 700;
    border-radius: 7px;
}

QPushButton#PrimaryButton:hover {
    background-color: #E86820;
}

QPushButton#PrimaryButton:pressed {
    background-color: #B3410B;
}

QPushButton#SuccessButton {
    background-color: #4A7A3A;
    color: #FFFFFF;
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    font-weight: 700;
    border-radius: 7px;
}

QPushButton#SuccessButton:hover {
    background-color: #5C9449;
}

QPushButton#DangerButton {
    background-color: #D94E4E;
    color: #FFFFFF;
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.25);
    font-weight: 700;
    border-radius: 7px;
}

QPushButton#DangerButton:hover {
    background-color: #EE6060;
}

/* Entradas de texto y números con efecto de profundidad hundida */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: rgba(22, 23, 24, 0.75);
    color: #ECEFF1;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 7px;
    padding: 6px 10px;
    selection-background-color: #D95C14;
    selection-color: #FFFFFF;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #D95C14;
}

/* Desplegables QComboBox Popups con vista QListView */
QComboBox QAbstractItemView,
QComboBox QListView,
QComboBox QFrame,
QComboBox QAbstractScrollArea {
    background-color: rgba(30, 32, 34, 0.95);
    color: #ECEFF1;
    border: 1px solid rgba(255, 255, 255, 0.15);
    outline: 0px;
    margin: 0px;
    padding: 0px;
    border-radius: 6px;
}

QComboBox QListView::item,
QComboBox QAbstractItemView::item {
    background-color: rgba(30, 32, 34, 0.95);
    color: #ECEFF1;
    min-height: 26px;
    padding: 4px 10px;
    border: none;
}

QComboBox QListView::item:hover,
QComboBox QListView::item:selected,
QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:selected {
    background-color: rgba(44, 46, 48, 0.95);
    color: #D95C14;
    font-weight: bold;
}

/* Pestañas (TabWidget & TabBar) estilo macOS Glassmorphism */
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: rgba(30, 32, 34, 0.60);
    border-radius: 12px;
}

QTabBar::tab {
    background-color: rgba(20, 22, 24, 0.70);
    color: #A0A6AC;
    padding: 8px 18px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: none;
    font-weight: bold;
    font-size: 12px;
}

QTabBar::tab:selected {
    background-color: rgba(30, 32, 34, 0.75);
    color: #D95C14;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-bottom: 2px solid #D95C14;
}

QTabBar::tab:hover:!selected {
    background-color: rgba(38, 41, 43, 0.80);
    color: #ECEFF1;
}

/* Tablas con fondo cristalino */
QTableWidget {
    background-color: rgba(18, 19, 20, 0.65);
    alternate-background-color: rgba(18, 19, 20, 0.65);
    gridline-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9px;
    color: #ECEFF1;
}

QTableWidget::item {
    background-color: transparent;
}

QTableCornerButton::section {
    background-color: rgba(30, 32, 34, 0.80);
    border: none;
}

QHeaderView {
    background-color: rgba(30, 32, 34, 0.80);
    border: none;
}

QHeaderView::section {
    background-color: rgba(30, 32, 34, 0.80);
    color: #F0A060;
    padding: 6px 8px;
    font-weight: bold;
    border: 1px solid rgba(18, 19, 20, 0.5);
}

QTableWidget::item:selected {
    background-color: rgba(63, 66, 69, 0.85);
    color: #D95C14;
}

/* GroupBox y Tarjetas con elevación vítrea */
QGroupBox {
    background-color: rgba(30, 32, 34, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: bold;
    color: #F0A060;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 2px 8px;
    background-color: rgba(20, 22, 24, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
}

/* CheckBox y RadioButton */
QCheckBox, QRadioButton {
    spacing: 8px;
    color: #ECEFF1;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #4A4D50;
    background-color: #161718;
}

QCheckBox::indicator:checked {
    background-color: #D95C14;
    border-color: #D95C14;
}

/* ScrollBars Vertical y Horizontal estilo macOS */
QScrollBar:vertical {
    border: none;
    background-color: #161718;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #4A4D50;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #606468;
}

QScrollBar:horizontal {
    border: none;
    background-color: #161718;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #4A4D50;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #606468;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
    background: none;
}

QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
}

/* QScrollArea y QListWidget */
QScrollArea {
    background-color: #161718;
    border: 1px solid #3F4245;
    border-radius: 8px;
}

QScrollArea > QWidget > QWidget {
    background-color: #161718;
}

QListWidget {
    background-color: #161718;
    border: 1px solid #3F4245;
    border-radius: 8px;
    color: #ECEFF1;
}

QListWidget::item {
    padding: 6px;
    border-radius: 5px;
}

QListWidget::item:selected {
    background-color: #2C2E30;
    color: #D95C14;
}
"""
