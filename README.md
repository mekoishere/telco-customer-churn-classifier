## main.py:
- Wczytuje dane z CSV
- Tworzy model, dobiera najlepsze jego parametry

### Zapisuje do plików .pkl: 
- Gotowy model 
- Kolumny danych - żeby później korzystać z nich w dobrej kolejności

## app.py:
- Wczytuje zasoby: Pobiera gotowy model i listę kolumn z plików .pkl
- Interfejs użytkownika: Wyświetla suwaki i pola do wpisania danych klienta
- Przetwarza dane: Automatycznie dopasowuje wpisane wartości do formatu, którego wymaga model
- Wyświetla wynik: Pokazuje prawdopodobieństwo odejścia i werdykt
- Loguje historię

## Użytkowanie:
- Uruchomić main.py
- main.py trenuje model i generuje pliki
- Przy pomocy `python -m streamlit run app.py` włączyć stronę
