{
  description = "RIFT-HARP basic development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python314;
    in {
      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          python
          uv
          git
          ffmpeg
          pkg-config
          ruff
          stdenv.cc.cc.lib
        ];

        UV_PYTHON = "${python}/bin/python";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
      };
    };
}
