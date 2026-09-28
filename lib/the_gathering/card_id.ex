defmodule TheGathering.CardId do
  @moduledoc """
  Locates the published card-recognition bundle that the webcam table loads in the browser.

  Bundles are built by `ml/` (`python -m cardid.export`) and copied to the server with
  `python -m cardid.publish <bundle> --to host:DATA_DIR/cardid`, which leaves this layout:

      DATA_DIR/cardid/<version>/{manifest.json,arts.json,detector.onnx,embed.onnx,search.onnx}
      DATA_DIR/cardid/current -> <version>

  The app never runs the models; it serves the files of the version `current` points at.

  `table_detector.onnx` (a dense multi-card box detector) may ship ahead of the embedding
  pipeline (`arts.json`/`detector.onnx`/`embed.onnx`/`search.onnx`), which identifies which
  card each box is; the browser degrades to boxes-only recognition until both exist.
  """

  @files ~w(manifest.json arts.json detector.onnx embed.onnx search.onnx printings.json table_detector.onnx)
  @version_pattern ~r/\A[A-Za-z0-9][A-Za-z0-9._-]*\z/

  @doc "Bundle file names the app serves, in the order the browser loads them."
  def files, do: @files

  @doc "The bundle root, `DATA_DIR/cardid`."
  def bundle_dir do
    Path.join(Application.fetch_env!(:the_gathering, :data_dir), "cardid")
  end

  @doc """
  The manifest of the bundle `current` points at, or `{:error, :not_found}` when nothing has
  been published. The version is read from the manifest, not the symlink, so a plain copied
  directory named `current` works too.
  """
  def current_manifest do
    with {:ok, contents} <- File.read(Path.join([bundle_dir(), "current", "manifest.json"])),
         {:ok, %{"version" => version} = manifest} when is_binary(version) <-
           Jason.decode(contents),
         true <- valid_version?(version) do
      {:ok, manifest}
    else
      _missing_or_invalid -> {:error, :not_found}
    end
  end

  @doc "Absolute path of a bundle file, refusing names and versions outside the bundle root."
  def file_path(version, name) when is_binary(version) and is_binary(name) do
    path = Path.join([bundle_dir(), version, name])

    if valid_version?(version) and name in @files and File.regular?(path) do
      {:ok, path}
    else
      {:error, :not_found}
    end
  end

  defp valid_version?(version),
    do: version != "current" and Regex.match?(@version_pattern, version)
end
