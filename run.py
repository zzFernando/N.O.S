#!/usr/bin/env python3
"""
N.O.S - Neuroevolution Analysis
Script de inicialização principal
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Menu principal do N.O.S"""
    print("🧠 N.O.S - Neuroevolution Analysis")
    print("=" * 50)
    print("1. 🚀 Iniciar Visualização")
    print("2. 🏋️  Executar Treinamento NEAT")
    print("3. 📊 Gerar Dados de Projeção")
    print("4. 🧪 Testar Visualização")
    print("5. 📖 Ver Documentação")
    print("0. ❌ Sair")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nEscolha uma opção: ").strip()
            
            if choice == "1":
                print("\n🚀 Iniciando visualização...")
                subprocess.run([sys.executable, "viz/launch.py"])
                break
                
            elif choice == "2":
                print("\n🏋️  Executando treinamento NEAT...")
                subprocess.run([sys.executable, "core/train.py"])
                break
                
            elif choice == "3":
                print("\n📊 Gerando dados de projeção...")
                subprocess.run([sys.executable, "viz/generate_data.py"])
                break
                
            elif choice == "4":
                print("\n🧪 Testando visualização...")
                subprocess.run([sys.executable, "-m", "http.server", "8000"], 
                             cwd="viz")
                break
                
            elif choice == "5":
                print("\n📖 Abrindo documentação...")
                print("README principal: README.md")
                print("Documentação da visualização: viz/README.md")
                break
                
            elif choice == "0":
                print("\n👋 Até logo!")
                break
                
            else:
                print("❌ Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main() 