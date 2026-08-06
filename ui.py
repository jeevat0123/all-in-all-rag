from functools import lru_cache
from nicegui import ui
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import uuid

# ==========================================
# 1. STATE & CACHING
# ==========================================

SAMPLE_TEXT = """The James Webb Space Telescope (JWST) is a space telescope
designed to conduct infrared astronomy. Its high-resolution and
high-sensitivity instruments allow it to view objects too old,
distant, or faint for the Hubble Space Telescope.

Quantum computing is a rapidly-emerging technology that
harnesses the laws of quantum mechanics to solve problems too
complex for classical computers. Today, IBM Quantum makes real
quantum hardware available to hundreds of thousands of
developers.

Photosynthesis is a process used by plants and other organisms
to convert light energy into chemical energy that can later be
released to fuel the organism's activities."""

@lru_cache(maxsize=3)
def get_embeddings(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Module-level state for single-session/demo runtime
current_chunks = []
current_ids = []
active_vectorstore = None


# ==========================================
# 2. PIPELINE REFRESH LOGIC
# ==========================================

def refresh_pipeline(e=None):
    """Executes the full pipeline reactively: Chunk -> Embed -> Index -> Plot -> Search."""
    global current_chunks, current_ids, active_vectorstore

    text = source_doc.value
    if not text:
        chunks_output.value = "Source document is empty."
        current_chunks = []
        current_ids = []
        plot_container.clear()
        search_results_container.clear()
        db_status.set_text("Status: Empty Source")
        return

    db_status.set_text("Status: Embedding...")

    # 1. Chunking Step
    strategy = split_strategy.value
    c_size = int(chunk_size.value or 150)
    c_overlap = int(overlap_size.value or 20)
    parsed_seps = (separators_input.value or "\n\n").replace('\\n', '\n')

    try:
        if strategy == 'Paragraph (Fixed Character)':
            splitter = CharacterTextSplitter(separator=parsed_seps, chunk_size=c_size, chunk_overlap=c_overlap, is_separator_regex=False)
            current_chunks = splitter.split_text(text)
        elif strategy == 'Recursive Character':
            splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap, is_separator_regex=False)
            current_chunks = splitter.split_text(text)
        elif strategy == 'Semantic Chunking':
            splitter = SemanticChunker(get_embeddings("all-MiniLM-L6-v2"), breakpoint_threshold_type="percentile")
            current_chunks = splitter.split_text(text)

        current_ids = [f"id_{i+1}" for i in range(len(current_chunks))]

        # Display Chunks Text
        formatted_chunks = [f"--- Chunk {i+1} ({len(c)} chars) ---\n{c}" for i, c in enumerate(current_chunks)]
        chunks_output.value = "\n\n".join(formatted_chunks)

    except Exception as err:
        chunks_output.value = f"Chunking Error: {err}"
        db_status.set_text("Status: Error")
        return

    # 2. Embedding & Vector DB Indexing Step
    if not current_chunks:
        db_status.set_text("Status: No chunks")
        return

    model_name = embed_model_select.value.split(" ")[0]
    embedder = get_embeddings(model_name)

    try:
        vectors = embedder.embed_documents(current_chunks)

        # Auto-reindex Chroma in-memory/ephemeral
        active_vectorstore = Chroma(
            collection_name="sandbox_collection",
            embedding_function=embedder,
            collection_metadata={"hnsw:space": "cosine"}
        )
        try:
            existing = active_vectorstore.get(include=[])
            if existing and existing['ids']:
                active_vectorstore.delete(ids=existing['ids'])
        except Exception as cleanup_err:
            print(f"[warn] Could not clear existing collection: {cleanup_err}")

        # Reuse pre-calculated vectors directly to avoid double-embedding overhead
        active_vectorstore._collection.upsert(
            documents=current_chunks,
            embeddings=vectors,
            ids=current_ids,
            metadatas=[{"id": cid} for cid in current_ids],
        )

        # Sync top-k bounds to available chunk count
        max_k = max(1, len(current_chunks))
        k_value_slider.props(f'max={max_k}')
        if k_value_slider.value and k_value_slider.value > len(current_chunks):
            k_value_slider.value = len(current_chunks)

        db_status.set_text(f"Status: Synced ({len(current_chunks)} vectors indexed)")

    except Exception as err:
        ui.notify(f"Embedding/Indexing Error: {err}", type='negative')
        db_status.set_text("Status: Error")
        return

    # 3. Visualization & Query Proximity Step
    query_text = (query_input.value or "").strip()
    nearest_indices: list[int] = []
    query_vector: list[float] | None = None
    try:
        if query_text:
            query_vector = embedder.embed_query(query_text)
            all_vectors = np.array(vectors + [query_vector])
        else:
            all_vectors = np.array(vectors)

        # PCA 2D Reduction
        explained_variance = None
        if len(all_vectors) >= 3:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(all_vectors)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))
        else:
            pca = PCA(n_components=1)
            coords_reduced = pca.fit_transform(all_vectors)
            coords_2d = np.hstack([coords_reduced, np.zeros((len(all_vectors), 1))])

        chunk_coords = coords_2d[:len(vectors)]
        x_vals = chunk_coords[:, 0].tolist()
        y_vals = chunk_coords[:, 1].tolist()

        # Alternate label positions to reduce overlap
        text_positions = ['top center' if i % 2 == 0 else 'bottom center' for i in range(len(current_chunks))]

        top_k = int(k_value_slider.value or 3)

        # Cosine distance computation matching vector store metric
        if query_text and query_vector is not None:
            distances = cosine_distances([query_vector], vectors)[0]
            nearest_indices = np.argsort(distances)[:min(top_k, len(vectors))]

        marker_colors = [
            '#f59e0b' if i in nearest_indices else '#2dd4bf'
            for i in range(len(current_chunks))
        ]

        data_traces = [{
            'type': 'scatter',
            'mode': 'markers+text',
            'x': x_vals,
            'y': y_vals,
            'text': [f"Chunk {i+1}" for i in range(len(current_chunks))],
            'textposition': text_positions,
            'marker': {'size': 14, 'color': marker_colors},
            'name': 'Chunks'
        }]

        # Optional Query Proximity Lines
        if query_text:
            query_coord = coords_2d[-1]

            data_traces.append({
                'type': 'scatter',
                'mode': 'markers+text',
                'x': [query_coord[0]],
                'y': [query_coord[1]],
                'text': ['Query'],
                'textposition': 'top center',
                'marker': {'size': 18, 'color': '#f472b6'},
                'name': 'Question'
            })

            line_x, line_y = [], []
            for idx in nearest_indices:
                line_x.extend([query_coord[0], x_vals[idx], None])
                line_y.extend([query_coord[1], y_vals[idx], None])

            data_traces.append({
                'type': 'scatter',
                'mode': 'lines',
                'x': line_x,
                'y': line_y,
                'line': {'color': '#9ca3af', 'dash': 'dash', 'width': 2},
                'showlegend': False,
                'hoverinfo': 'none'
            })

        title_text = 'Live Vector Space & Query Proximity'
        if explained_variance is not None:
            title_text += f'  (PCA: {explained_variance*100:.0f}% variance explained)'

        fig = {
            'data': data_traces,
            'layout': {
                'title': {'text': title_text, 'font': {'size': 13}, 'y': 0.95},
                'margin': {'l': 30, 'r': 30, 't': 50, 'b': 30},
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'legend': {'orientation': 'h', 'y': 1.1, 'x': 0.5, 'xanchor': 'center'}
            }
        }

        plot_container.clear()
        with plot_container:
            ui.plotly(fig).classes('w-full h-[320px]')

        # 4. Live Search Execution
        search_results_container.clear()
        if query_text and active_vectorstore:
            results = active_vectorstore.similarity_search_with_score(query_text, k=int(k_value_slider.value or 3))
            with search_results_container:
                for doc, score in results:
                    with ui.card().classes('w-full p-2 bg-gray-50 border shadow-none mb-1'):
                        ui.label(doc.page_content).classes('text-xs text-gray-800')
                        ui.label(f"Score: {score:.4f} | ID: {doc.metadata.get('id')}").classes('text-[10px] text-gray-400')
        else:
            with search_results_container:
                ui.label("Enter a query above to see live vector search results.").classes('text-xs text-gray-400 italic')

    except Exception as err:
        ui.notify(f"Plot/Search Error: {err}", type='negative')
        db_status.set_text(f"Status: Synced ({len(current_chunks)} vectors indexed) — plot/search failed")


# def add_custom_text():
#     """Append a user-entered snippet directly into the live index and re-render."""
#     global current_ids
#     text = (new_item_input.value or "").strip()
#     if not text or not active_vectorstore:
#         return
#     new_id = f"id_c_{uuid.uuid4().hex[:4]}"
#     active_vectorstore.add_texts(texts=[text], ids=[new_id], metadatas=[{"id": new_id}])
#     current_chunks.append(text)
#     current_ids.append(new_id)
#     new_item_input.value = ""
#     refresh_pipeline()


def reset_to_sample():
    """Restore default configurations and source text."""
    source_doc.value = SAMPLE_TEXT
    split_strategy.value = 'Recursive Character'
    chunk_size.value = 150
    overlap_size.value = 20
    separators_input.value = ''
    embed_model_select.value = 'all-MiniLM-L6-v2 (384d)'
    query_input.value = 'What is James Webb?'
    k_value_slider.value = 2
    refresh_pipeline()


# ==========================================
# 3. UI LAYOUT (RESPONSIVE 3-COLUMN SANDBOX)
# ==========================================

ui.add_head_html('<style>textarea { font-family: monospace; }</style>')

with ui.column().classes('w-full max-w-[95vw] mx-auto p-4 gap-4'):

    with ui.row().classes('w-full items-center justify-between'):
        ui.label('⚡ All-in-One Live RAG Sandbox').classes('text-2xl font-bold font-mono text-gray-800')
        ui.button('Reset to sample', on_click=reset_to_sample).props('outline').classes('text-xs')

    # Main Workspace Grid
    with ui.grid(columns=3).classes('w-full gap-4 items-start'):

        # COLUMN 1: CONFIG & SOURCE TEXT
        with ui.card().classes('w-full p-4 gap-4 bg-gray-50 border rounded-2xl'):
            ui.label('1. Source & Chunk Config').classes('font-bold text-gray-700')

            source_doc = ui.textarea(value=SAMPLE_TEXT, on_change=refresh_pipeline).classes('w-full text-xs').props('outlined autogrow')

            split_strategy = ui.select(
                ['Recursive Character', 'Paragraph (Fixed Character)', 'Semantic Chunking'],
                value='Recursive Character', label='Split Strategy', on_change=refresh_pipeline
            ).classes('w-full font-mono text-xs')

            with ui.row().classes('w-full gap-2'):
                chunk_size = ui.number(value=150, step=10, label='Size', on_change=refresh_pipeline).classes('flex-1 font-mono text-xs')
                overlap_size = ui.number(value=20, step=5, label='Overlap', on_change=refresh_pipeline).classes('flex-1 font-mono text-xs')

            separators_input = ui.input(placeholder='\\n\\n', label='Separators', on_change=refresh_pipeline).classes('w-full font-mono text-xs')

            embed_model_select = ui.select(
                ['all-MiniLM-L6-v2 (384d)', 'all-mpnet-base-v2 (768d)'],
                value='all-MiniLM-L6-v2 (384d)', label='Embedding Model', on_change=refresh_pipeline
            ).classes('w-full font-mono text-xs')

        # COLUMN 2: LIVE CHUNKS & DB FEED
        with ui.card().classes('w-full p-4 gap-4 bg-gray-50 border rounded-2xl'):
            ui.label('2. Live Chunks & DB State').classes('font-bold text-gray-700')

            chunks_output = ui.textarea().classes('w-full text-xs bg-white').props('outlined readonly autogrow')

            db_status = ui.label('Status: Initializing...').classes('text-xs font-semibold text-teal-600')

            # with ui.row().classes('w-full gap-2 items-center'):
            #     new_item_input = ui.input(placeholder='Append custom text...').classes('flex-1 text-xs')
            #     ui.button('Add', on_click=add_custom_text).classes('bg-indigo-600 text-white text-xs')

        # COLUMN 3: VECTOR SPACE & RETRIEVAL
        with ui.card().classes('w-full p-4 gap-4 bg-gray-50 border rounded-2xl'):
            ui.label('3. Vector Space & Retrieval').classes('font-bold text-gray-700')

            with ui.row().classes('w-full gap-2'):
                query_input = ui.input(value='What is James Webb?', label='Query Question', on_change=refresh_pipeline).classes('flex-1 text-xs')
                k_value_slider = ui.number(value=2, min=1, max=5, label='Top-k', on_change=refresh_pipeline).classes('w-16 text-xs')

            plot_container = ui.column().classes('w-full bg-white rounded-xl border p-1')
            with plot_container:
                ui.label("Loading graph...").classes('text-xs text-gray-400 italic p-10 text-center w-full')

            ui.label('Retrieved Matches:').classes('text-xs font-bold text-gray-600 mt-2')
            search_results_container = ui.column().classes('w-full max-h-[300px] overflow-y-auto gap-1')


# Run initial pipeline render
refresh_pipeline()

ui.run(title="All-in-One Live RAG Sandbox")