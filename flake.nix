{
  description = "nix develop environment for radioML/dataset";

  # nixConfig = {
  #   extra-substituters = [
  #     "https://mirrors.tuna.tsinghua.edu.cn/nix-channels/store?priority=30"
  #     "https://nix-community.cachix.org?priority=40"
  #     "https://cache.nixos.org?priority=50"
  #   ];
  #   extra-trusted-public-keys = [
  #     "cache.nixos.org-1:6NCHdD59X431o0gW2vTwlnWa6G9qyQ/8LokvxsP2W1c="
  #     "nix-community.cachix.org-1:mB9lNPq1U9ZqC1jmv1s2+3eu1i6jJ3r5C6hDXWjY3jA="
  #   ];
  # };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
    };
    pythonEnv = pkgs.python313.withPackages (ps:
      with ps; [
      numpy
      scipy
      h5py
      pytest
      tqdm
      ]);
    gnuradioPythonPath = "${pkgs.gnuradioMinimal}/${pkgs.gnuradioMinimal.python.sitePackages}";
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        gnuradio
        pythonEnv
      ];
      shellHook = ''
        export PYTHONPATH="${gnuradioPythonPath}:$PWD:$PYTHONPATH"
      '';
    };
    formatter.${system} = pkgs.alejandra;
  };
}
