# mem0-mastermind-test

This project is an experimental **AI-powered Mastermind game** that uses **OpenAI models (e.g., GPT-4o)** with the **mem0 memory system** and a **Qdrant vector database** to simulate reasoning and memory during gameplay.

---

## Getting Started

### 1. Start the Qdrant container

Before running the project, you need to start a Qdrant instance using Docker.

For **PowerShell (Windows)**:

```powershell
docker run --name qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v "E:\study\CAPSTONE\qdrant_data:/qdrant/storage" `
  -e QDRANT__SERVICE__GRPC_PORT=6334 `
  qdrant/qdrant:latest
```

**Explanation:**

* `-p 6333:6333`: exposes Qdrant’s REST API port
* `-p 6334:6334`: exposes the gRPC service port
* `-v`: mounts a local folder to persist Qdrant data
* `QDRANT__SERVICE__GRPC_PORT`: configures the gRPC port
* Once running, Qdrant will be accessible at [http://localhost:6333](http://localhost:6333)

---

### 2. Clone the repository

```bash
git clone https://github.com/Dante1028/mem0-mastermind-test.git
cd mem0-mastermind-test/mastermind
```

---

### 3. Set your OpenAI API key

You’ll need an OpenAI API key to run the game.

For **PowerShell (Windows)**:

```powershell
$env:OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
```

For **macOS / Linux**:

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
```

---

### 4. Install dependencies

It’s recommended to use a Python virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate     # (Windows)
# or
source venv/bin/activate    # (macOS / Linux)

pip install -r requirements.txt
```

---

### 5. Run the full game

Launch the game with:

```bash
python run_full_game.py --model_type mem0_openai --model gpt-4o --num_runs 2 --code_length 4 --num_colors 6
```

#### Command-line arguments:

| Argument        | Description                                  | Default  |
| --------------- | -------------------------------------------- | -------- |
| `--model_type`  | Type of model interface (e.g. `mem0_openai`) | required |
| `--model`       | OpenAI model name (e.g. `gpt-4o`)            | required |
| `--num_runs`    | Number of simulated runs                     | 2        |
| `--code_length` | Length of the target code                    | 4        |
| `--num_colors`  | Number of possible colors                    | 6        |

---

## Project Structure

```
mem0-mastermind-test/
├── mastermind/
│   ├── run_full_game.py         # Main game runner
│   ├── mastermind_game.py       # Core game logic
│   ├── memory_manager.py        # mem0 memory handler
│   ├── config.py                # Configurations (e.g. Qdrant setup)
│   └── ...
└── requirements.txt
```

---

## Verify Qdrant

Check that Qdrant is running:

* Open in browser: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
* Or run:

  ```bash
  curl http://localhost:6333/collections
  ```

---

## Requirements

* Python 3.9+
* Docker
* OpenAI API key
* Qdrant (latest)

---

## Example Output

When you run the script, the model will:

* Generate guesses for the secret code
* Receive feedback (correct color / correct position)
* Use mem0 memory to adjust its reasoning over multiple runs

