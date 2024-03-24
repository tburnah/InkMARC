using System.Collections.Specialized;
using OcuInkTrain.Extensions;
using Microsoft.Maui.Handlers;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;
using OcuInkTrain.Views;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingView handler
/// </summary>
public class AdvancedDrawingViewHandler : ViewHandler<IOcuInkDrawingView, OcuInkDrawingView>, IAdvancedDrawingViewHandler
{
    /// <summary>
    /// <see cref ="PropertyMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly IPropertyMapper<IOcuInkDrawingView, AdvancedDrawingViewHandler> DrawingViewMapper = new PropertyMapper<IOcuInkDrawingView, AdvancedDrawingViewHandler>(ViewMapper)
    {
        [nameof(IOcuInkDrawingView.DrawAction)] = MapDrawAction,
        [nameof(IOcuInkDrawingView.ShouldClearOnFinish)] = MapShouldClearOnFinish,
        [nameof(IOcuInkDrawingView.IsMultiLineModeEnabled)] = MapIsMultiLineModeEnabled,
        [nameof(IOcuInkDrawingView.LineColor)] = MapLineColor,
        [nameof(IOcuInkDrawingView.LineWidth)] = MapLineWidth,
        [nameof(IOcuInkDrawingView.Background)] = MapDrawingViewBackground,
        [nameof(IOcuInkDrawingView.Lines)] = MapLines, // `IOcuInkDrawingView.Lines` must be mapped last
    };


    /// <summary>
    /// <see cref ="CommandMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly CommandMapper<IOcuInkDrawingView, AdvancedDrawingViewHandler> DrawingViewCommandMapper = new(ViewCommandMapper);

    /// <summary>
    /// Initialize new instance of <see cref="AdvancedDrawingViewHandler"/>.
    /// </summary>
    /// <param name="mapper">Custom instance of <see cref="PropertyMapper"/>, if it's null the <see cref="DrawingViewMapper"/> will be used</param>
    /// <param name="commandMapper">Custom instance of <see cref="CommandMapper"/></param>
    public AdvancedDrawingViewHandler(IPropertyMapper? mapper, CommandMapper? commandMapper)
        : base(mapper ?? DrawingViewMapper, commandMapper ?? DrawingViewCommandMapper)
    {

    }

    /// <summary>
    /// Initialize new instance of <see cref="AdvancedDrawingViewHandler"/>.
    /// </summary>
    public AdvancedDrawingViewHandler() : this(DrawingViewMapper, DrawingViewCommandMapper)
    {
    }

    IAdvancedDrawingLineAdapter adapter = new AdvancedDrawingLineAdapter();

    /// <inheritdoc />
    public void SetDrawingLineAdapter(IAdvancedDrawingLineAdapter drawingLineAdapter)
    {
        adapter = drawingLineAdapter;
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.Lines"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapLines(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetLines(view);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.ShouldClearOnFinish"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapShouldClearOnFinish(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetShouldClearOnFinish(view.ShouldClearOnFinish);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.LineColor"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapLineColor(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetLineColor(view.LineColor);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.LineWidth"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapLineWidth(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetLineWidth(view.LineWidth);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.IsMultiLineModeEnabled"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapIsMultiLineModeEnabled(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetIsMultiLineModeEnabled(view.IsMultiLineModeEnabled);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IOcuInkDrawingView.DrawAction"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapDrawAction(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetDrawAction(view.DrawAction);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView Background property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IOcuInkDrawingView"/>.</param>
    public static void MapDrawingViewBackground(AdvancedDrawingViewHandler handler, IOcuInkDrawingView view)
    {
        handler.PlatformView.SetPaint(view.Background ?? new SolidPaint(CommunityToolkit.Maui.Core.DrawingViewDefaults.BackgroundColor));
    }

    /// <inheritdoc />
    protected override void ConnectHandler(OcuInkDrawingView platformView)
    {
        base.ConnectHandler(platformView);
        platformView.Initialize();

        platformView.DrawingStarted += OnPlatformViewDrawingStarted;
        platformView.DrawingCancelled += OnPlatformViewDrawingCancelled;
        platformView.Drawing += OnPlatformViewDrawing;
        platformView.DrawingLineCompleted += OnPlatformViewDrawingLineCompleted;
        VirtualView.Lines.CollectionChanged += OnVirtualViewLinesCollectionChanged;
        platformView.Lines.CollectionChanged += OnPlatformViewLinesCollectionChanged;
    }

    /// <inheritdoc />
    protected override void DisconnectHandler(OcuInkDrawingView platformView)
    {
        platformView.DrawingStarted -= OnPlatformViewDrawingStarted;
        platformView.DrawingCancelled -= OnPlatformViewDrawingCancelled;
        platformView.Drawing -= OnPlatformViewDrawing;
        platformView.DrawingLineCompleted -= OnPlatformViewDrawingLineCompleted;
        VirtualView.Lines.CollectionChanged -= OnVirtualViewLinesCollectionChanged;
        platformView.Lines.CollectionChanged -= OnPlatformViewLinesCollectionChanged;

        platformView.CleanUp();

        base.DisconnectHandler(platformView);
    }

    /// <inheritdoc />
#if ANDROID
	protected override OcuInkDrawingView CreatePlatformView() => new(Context);
#else
    protected override OcuInkDrawingView CreatePlatformView() => new();
#endif

    void OnPlatformViewDrawingLineCompleted(object? sender, OcuInkDrawingLineCompletedEventArgs e)
    {
        var drawingLine = adapter.ConvertOcuInkDrawingLine(e.Line);
        VirtualView.OnDrawingLineCompleted(drawingLine);
    }

    void OnPlatformViewDrawing(object? sender, OcuInkOnDrawingEventArgs e)
    {
        VirtualView.OnPointDrawn(e.Point);
    }

    void OnPlatformViewDrawingStarted(object? sender, OcuInkDrawingStartedEventArgs e)
    {
        VirtualView.OnDrawingLineStarted(e.Point);
    }

    void OnPlatformViewDrawingCancelled(object? sender, EventArgs e)
    {
        VirtualView.OnDrawingLineCancelled();
    }

    void OnVirtualViewLinesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        PlatformView.Lines.CollectionChanged -= OnPlatformViewLinesCollectionChanged;
        PlatformView.SetLines(VirtualView);
        PlatformView.Lines.CollectionChanged += OnPlatformViewLinesCollectionChanged;
    }

    void OnPlatformViewLinesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        VirtualView.Lines.CollectionChanged -= OnVirtualViewLinesCollectionChanged;
        VirtualView.SetLines(PlatformView, adapter);
        VirtualView.Lines.CollectionChanged += OnVirtualViewLinesCollectionChanged;
    }
}
