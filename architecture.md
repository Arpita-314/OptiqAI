<!-- filepath: [architecture.md](http://_vscodecontentref_/0) -->
```mermaid
flowchart TB
    subgraph User_Layer["User Layer"]
        A[GUI\nQt/PySide] -->|Drag & Drop| B[Web Dashboard\nPlotly/Dash]
    end

    subgraph Core_Engine["Core Engine"]
        B --> C[Simulation Service\nPython/C++]
        C --> D[FFT Solver\nCUDA/pyFFTW]
        C --> E[Angular Spectrum\nOpenCL]
    end

    subgraph AI_Layer["AI Layer"]
        D --> F[AI Trainer\nPyTorch]
        E --> F
        F --> G[Phase Retrieval CNN\nONNX]
    end

    subgraph Data_Layer["Data Layer"]
        G --> H[Data Pipeline\nApache Kafka]
        H --> I[Storage\nHDF5/S3]
    end

    subgraph Output_Layer["Output Layer"]
        I --> J[Fabrication Export\nGDSII]
        I --> K[Visualization\nVTK/Three.js]
    end

    style User_Layer fill:#f9f9f9,stroke:#666
    style Core_Engine fill:#e6f7ff,stroke:#1890ff
    style AI_Layer fill:#fff2e6,stroke:#fa8c16
    style Data_Layer fill:#f6ffed,stroke:#52c41a
    style Output_Layer fill:#f0f5ff,stroke:#1d39c4