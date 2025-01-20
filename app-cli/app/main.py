import typer
import docker
import requests
import os
from typing import Optional
from rich import print
from rich.console import Console
from getpass import getpass
import subprocess

app = typer.Typer()
console = Console()

def check_service(url: str) -> bool:
    try:
        requests.get(url, timeout=5)
        return True
    except:
        return False

@app.command()
def start(all: bool = typer.Option(False, "--all", help="Start all services")):
    try:
        client = docker.from_env()
        if all:
            subprocess.run(["docker-compose", "up", "-d"])
            console.print("Started all services", style="green")
        else:
            subprocess.run(["docker-compose", "up", "-d", "db", "storage"])
            console.print("Started minimal backend services", style="green")
    except Exception as e:
        console.print(f"Error starting services: {str(e)}", style="red")

@app.command()
def start_ui():
    try:
        subprocess.run(["docker-compose", "up", "-d", "frontend"])
        console.print("Started frontend service", style="green")
    except Exception as e:
        console.print(f"Error starting frontend: {str(e)}", style="red")

@app.command()
def create(
    name: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    bio: Optional[str] = None
):
    if not all([name, username, password, bio]):
        name = typer.prompt("Enter name")
        username = typer.prompt("Enter username")
        password = typer.prompt("Enter password", hide_input=True)
        bio = typer.prompt("Enter bio")
    
    try:
        # POST to backend API
        response = requests.post(
            "http://localhost:8000/users",
            json={"name": name, "username": username, "password": password, "bio": bio}
        )
        console.print("User created successfully", style="green")
    except Exception as e:
        console.print(f"Error creating user: {str(e)}", style="red")

@app.command()
def scan(path: str = typer.Argument(..., help="Directory path to scan")):
    try:
        entries = os.listdir(path)
        for entry in entries:
            full_path = os.path.join(path, entry)
            try:
                response = requests.post(
                    "http://localhost:8000/entries",
                    json={"path": full_path}
                )
                console.print(f"Posted entry: {entry}", style="green")
            except Exception as e:
                console.print(f"Error posting entry {entry}: {str(e)}", style="red")
    except Exception as e:
        console.print(f"Error scanning directory: {str(e)}", style="red")

@app.command()
def status():
    services = {
        "Backend": "http://localhost:8000/health",
        "Frontend": "http://localhost:3000",
        "Database": "http://localhost:5432"
    }
    
    for service, url in services.items():
        status = "Running" if check_service(url) else "Stopped"
        color = "green" if status == "Running" else "red"
        console.print(f"{service}: {status}", style=color)

@app.callback()
def main():
    """
    CLI for managing application services and users.
    
    Commands:
    start: Start backend services
    start-ui: Start frontend service
    create: Create new user
    scan: Scan directory
    status: Check services status
    """

if __name__ == "__main__":
    app()