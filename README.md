<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC FloodNet: Urban Flash-Flood Forecasting</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }

        .uva-gradient {
            background: linear-gradient(135deg, #232D4B 0%, #2c3e50 100%);
        }

        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s ease;
        }

        .card:hover {
            transform: translateY(-2px);
        }

        .badge-blue {
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .step-number {
            width: 32px;
            height: 32px;
            background-color: #232D4B;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }

        code {
            background-color: #f1f5f9;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
        }

        .accent-border {
            border-left: 4px solid #E57200; /* UVA Orange */
        }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-5xl mx-auto">
        <!-- Hero Header -->
        <header class="uva-gradient text-white p-8 md:p-12 rounded-3xl mb-8 shadow-2xl relative overflow-hidden">
            <div class="relative z-10">
                <div class="flex items-center gap-2 mb-4 opacity-90">
                    <i data-lucide="waves" class="w-6 h-6"></i>
                    <span class="font-semibold tracking-wider uppercase text-sm">Deep Learning Research</span>
                </div>
                <h1 class="text-4xl md:text-5xl font-bold mb-4">NYC FloodNet: Urban Flash-Flood Forecasting</h1>
                <p class="text-xl opacity-90 font-light max-w-2xl mb-6">
                    Leveraging high-frequency ultrasonic sensor data and HRRR meteorological forcing to predict street-level flood depths.
                </p>
                <div class="flex flex-wrap gap-4 items-center text-sm">
                    <div class="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-md">
                        <i data-lucide="graduation-cap" class="w-4 h-4 text-orange-400"></i>
                        <span>University of Virginia</span>
                    </div>
                    <div class="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-md">
                        <i data-lucide="database" class="w-4 h-4 text-blue-400"></i>
                        <span>Hydroinformatics Final Project</span>
                    </div>
                </div>
            </div>
            <!-- Decorative Wave SVG -->
            <svg class="absolute bottom-0 right-0 w-1/2 opacity-10" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <path fill="#FFFFFF" d="M44.7,-76.4C58.1,-69.2,69.2,-58.1,76.4,-44.7C83.7,-31.3,87.1,-15.7,85.3,-0.9C83.6,13.8,76.7,27.7,67.7,40.1C58.7,52.5,47.6,63.5,34.4,70.9C21.2,78.3,5.9,82.1,-10.1,80.4C-26.1,78.7,-42.8,71.5,-55.8,60.6C-68.8,49.7,-78.1,35.1,-82.4,19.3C-86.7,3.5,-85.9,-13.5,-79.8,-28.6C-73.7,-43.7,-62.3,-56.8,-48.9,-64.1C-35.5,-71.4,-20.1,-72.9,-2.4,-68.8C15.3,-64.7,31.3,-83.6,44.7,-76.4Z" transform="translate(100 100)" />
            </svg>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <!-- Left Column: Data & Engineering -->
            <div class="md:col-span-2 space-y-8">
                <!-- Dataset Architecture -->
                <section class="card p-6">
                    <div class="flex items-center gap-2 mb-6">
                        <i data-lucide="folder-tree" class="text-blue-600"></i>
                        <h2 class="text-2xl font-bold">Dataset Architecture</h2>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left">
                            <thead class="border-b border-slate-100">
                                <tr class="text-slate-500 text-sm">
                                    <th class="pb-3 font-medium">Filename</th>
                                    <th class="pb-3 font-medium">Size</th>
                                    <th class="pb-3 font-medium">Description</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-50">
                                <tr>
                                    <td class="py-4 font-mono text-sm text-blue-700">delineated_storms.parquet</td>
                                    <td class="py-4"><span class="badge-blue">353 MB</span></td>
                                    <td class="py-4 text-sm text-slate-600">Primary modeling set (6h IETD).</td>
                                </tr>
                                <tr>
                                    <td class="py-4 font-mono text-sm text-blue-700">floodnet_full_merged.parquet</td>
                                    <td class="py-4"><span class="badge-blue">2.63 GB</span></td>
                                    <td class="py-4 text-sm text-slate-600">Master join (Sensor depths + Weather features).</td>
                                </tr>
                                <tr>
                                    <td class="py-4 font-mono text-sm text-blue-700">nyc_precip_master.parquet</td>
                                    <td class="py-4"><span class="badge-blue">104 MB</span></td>
                                    <td class="py-4 text-sm text-slate-600">Cleaned hourly & max-intensity precip.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>

                <!-- Engineering Highlights -->
                <section class="card p-6 accent-border">
                    <div class="flex items-center gap-2 mb-6 text-orange-700">
                        <i data-lucide="settings" class="w-6 h-6"></i>
                        <h2 class="text-2xl font-bold">Data Engineering Highlights</h2>
                    </div>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                        <div class="space-y-2">
                            <h3 class="font-semibold text-slate-800 flex items-center gap-2">
                                <i data-lucide="scissors" class="w-4 h-4"></i> Storm Delineation
                            </h3>
                            <p class="text-sm text-slate-600">Events isolated using a 6-hour dry-gap criterion (MIT standard).</p>
                        </div>
                        <div class="space-y-2">
                            <h3 class="font-semibold text-slate-800 flex items-center gap-2">
                                <i data-lucide="clock" class="w-4 h-4"></i> Lead/Lag Buffering
                            </h3>
                            <p class="text-sm text-slate-600">Includes 2h antecedent moisture lead-up and 6h flood recession limb.</p>
                        </div>
                        <div class="space-y-2">
                            <h3 class="font-semibold text-slate-800 flex items-center gap-2">
                                <i data-lucide="shield-check" class="w-4 h-4"></i> Temporal Integrity
                            </h3>
                            <p class="text-sm text-slate-600">70/15/15 splits performed at the <strong>Storm ID level</strong> to prevent data leakage.</p>
                        </div>
                        <div class="space-y-2">
                            <h3 class="font-semibold text-slate-800 flex items-center gap-2">
                                <i data-lucide="maximize" class="w-4 h-4"></i> Compression
                            </h3>
                            <p class="text-sm text-slate-600">Utilizes Apache Parquet with ZSTD for high-speed I/O and low disk footprint.</p>
                        </div>
                    </div>
                </section>
            </div>

            <!-- Right Column: Pipeline & Guide -->
            <div class="space-y-8">
                <!-- Modeling Pipeline -->
                <section class="card p-6 bg-slate-900 text-white">
                    <div class="flex items-center gap-2 mb-6">
                        <i data-lucide="cpu" class="text-blue-400"></i>
                        <h2 class="text-xl font-bold">Modeling Pipeline</h2>
                    </div>
                    <div class="space-y-4">
                        <div class="border-l-2 border-blue-500 pl-4 py-1">
                            <h4 class="text-sm font-semibold text-blue-300 uppercase tracking-wider">Baseline</h4>
                            <p class="text-sm">Log-Ridge Regression</p>
                        </div>
                        <div class="border-l-2 border-purple-500 pl-4 py-1">
                            <h4 class="text-sm font-semibold text-purple-300 uppercase tracking-wider">SOTA ANN</h4>
                            <p class="text-sm">Res-ANN with LayerNorm</p>
                        </div>
                        <div class="border-l-2 border-cyan-500 pl-4 py-1">
                            <h4 class="text-sm font-semibold text-cyan-300 uppercase tracking-wider">SOTA RNN</h4>
                            <p class="text-sm">Attention-Bi-LSTM</p>
                        </div>
                    </div>
                    
                    <div class="mt-8 pt-6 border-t border-slate-700">
                        <h3 class="text-xs font-bold text-slate-400 uppercase mb-4">Hardware Optimization</h3>
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs">ANN Batch</span>
                            <span class="text-xs font-mono">32,768</span>
                        </div>
                        <div class="flex justify-between items-center mb-4">
                            <span class="text-xs">LSTM Batch</span>
                            <span class="text-xs font-mono">2,048</span>
                        </div>
                        <div class="bg-blue-500/10 text-blue-400 text-[10px] p-2 rounded border border-blue-500/20 flex items-center gap-2">
                            <i data-lucide="zap" class="w-3 h-3"></i>
                            Mixed Precision (torch.amp) Enabled
                        </div>
                    </div>
                </section>

                <!-- Reproducibility -->
                <section class="card p-6">
                    <h2 class="text-xl font-bold mb-6">Reproducibility</h2>
                    <div class="space-y-6">
                        <div class="flex gap-4">
                            <div class="step-number">1</div>
                            <div>
                                <h4 class="font-semibold text-sm">Setup</h4>
                                <p class="text-xs text-slate-500">Python 3.11+ environment with PyTorch and Optuna.</p>
                            </div>
                        </div>
                        <div class="flex gap-4">
                            <div class="step-number">2</div>
                            <div>
                                <h4 class="font-semibold text-sm">ETL</h4>
                                <p class="text-xs text-slate-500">Run DuckDB script to regenerate delineated storms.</p>
                            </div>
                        </div>
                        <div class="flex gap-4">
                            <div class="step-number">3</div>
                            <div>
                                <h4 class="font-semibold text-sm">Tune</h4>
                                <p class="text-xs text-slate-500">Execute Optuna Shootout for hyperparameter tuning.</p>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </div>

        <!-- Footer / Checksums -->
        <footer class="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="card p-4 bg-slate-50 text-slate-500 text-xs font-mono space-y-1">
                <div class="flex justify-between">
                    <span>delineated_storms.md5:</span>
                    <span class="text-slate-800">[PASTE_MD5_HERE]</span>
                </div>
                <div class="flex justify-between">
                    <span>floodnet_full.md5:</span>
                    <span class="text-slate-800">[PASTE_MD5_HERE]</span>
                </div>
            </div>
            <div class="flex items-center justify-center md:justify-end gap-6 text-slate-400">
                <div class="text-right">
                    <p class="text-xs font-bold text-slate-500">CONTACT</p>
                    <p class="text-sm">Michael Dunlap</p>
                </div>
                <div class="h-8 w-px bg-slate-200"></div>
                <div class="text-right">
                    <p class="text-xs font-bold text-slate-500">LICENSE</p>
                    <p class="text-sm">MIT / NYC FloodNet</p>
                </div>
            </div>
        </footer>
    </div>

    <script>
        // Initialize Lucide icons
        lucide.createIcons();
    </script>
</body>
</html>