from matplotlib import colors


class ColormapConfig:
    def __init__(self):
        self.cmap = None
        self.norm = None
        self.clevs = None
        self.bounds = None

        self.build_colormap()

    def build_colormap(self):
        # Define the colormap boundaries and colors

        # Eric's custom colormap
        # color_list = [
        #     "#4cecec",
        #     "#44c6f0",
        #     "#429afb",
        #     "#3431fd",
        #     "#40f600",
        #     "#3ada0b",
        #     "#2eb612",
        #     "#2a8a0f",
        #     "#f8f915",
        #     "#e9d11c",
        #     "#dcb11f",
        #     "#bd751f",
        #     "#f39a9c",
        #     "#f23a43",
        #     "#da1622",
        #     "#a90c1b",
        #     "#fa31ff",
        #     "#d32ada",
        #     "#9f1fa3",
        #     "#751678",
        #     "#ffffff",
        #     "#c1bdff",
        #     "#c5ffff",
        #     "#fcfec0",
        #     "#fcfec0"
        # ]

        # self.clevs = [1, 6.35, 12.7, 19.05, 25.4, 31.75, 38.1, 44.45, 50.8, 57.15, 63.5, 69.85,
        # 76.2, 82.55, 88.9, 95.25, 101.6, 114.3, 127.0, 139.7, 152.4, 165.1, 177.8, 190.5, 203.2]
        # self.bounds = [1, 12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 127.0, 152.4, 177.8, 203.2]

        # Shortened version of Eric's custom colormap
        color_list = [
            "#44c6f0",
            "#429afb",
            "#3431fd",
            "#f8f915",
            "#dcb11f",
            "#bd751f",
            "#f23a43",
            "#da1622",
            "#a90c1b",
        ]

        self.clevs = [1, 6.35, 12.7, 19.05, 25.4, 31.75, 38.1, 44.45, 50.8]  # 9
        self.bounds = [1, 12.7, 25.4, 38.1, 50.8]

        self.cmap = colors.ListedColormap(color_list)
        # self.cmap.set_over("#fcfec0") # from Eric's custom colormap
        self.cmap.set_over("darkmagenta")
        self.cmap.set_under("none")
        self.cmap.set_bad("gray", alpha=0.5)
        self.norm = colors.BoundaryNorm(self.clevs, self.cmap.N)
        self.cmap.name = "Custom Colormap"
