using System.Collections.Specialized;
using InkMARCDeform.Extensions;
using Microsoft.Maui.Handlers;
using InkMARCDeform.Interfaces;
using InkMARCDeform.Primatives;
using InkMARCDeform.Views;

namespace InkMARCDeform.Handlers;

/// <summary>
/// DrawingView handler
/// </summary>
public class AdvancedDrawingViewHandler : ViewHandler<IInkMARCDrawingView, InkMARCDrawingView>, IAdvancedDrawingViewHandler
{
    /// <summary>
    /// <see cref ="PropertyMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly IPropertyMapper<IInkMARCDrawingView, AdvancedDrawingViewHandler> DrawingViewMapper = new PropertyMapper<IInkMARCDrawingView, AdvancedDrawingViewHandler>(ViewMapper)
    {
        [nameof(IInkMARCDrawingView.DrawAction)] = MapDrawAction,
        [nameof(IInkMARCDrawingView.ShouldClearOnFinish)] = MapShouldClearOnFinish,
        [nameof(IInkMARCDrawingView.IsMultiLineModeEnabled)] = MapIsMultiLineModeEnabled,
        [nameof(IInkMARCDrawingView.LineColor)] = MapLineColor,
        [nameof(IInkMARCDrawingView.LineWidth)] = MapLineWidth,
        [nameof(IInkMARCDrawingView.Background)] = MapDrawingViewBackground,
        [nameof(IInkMARCDrawingView.Lines)] = MapLines, // `IInkMARCDrawingView.Lines` must be mapped last
    };


    /// <summary>
    /// <see cref ="CommandMapper"/> for DrawingView Control.
    /// </summary>
    public static readonly CommandMapper<IInkMARCDrawingView, AdvancedDrawingViewHandler> DrawingViewCommandMapper = new(ViewCommandMapper);

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
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.Lines"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapLines(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetLines(view);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.ShouldClearOnFinish"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapShouldClearOnFinish(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetShouldClearOnFinish(view.ShouldClearOnFinish);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.LineColor"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapLineColor(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetLineColor(view.LineColor);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.LineWidth"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapLineWidth(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetLineWidth(view.LineWidth);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.IsMultiLineModeEnabled"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapIsMultiLineModeEnabled(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetIsMultiLineModeEnabled(view.IsMultiLineModeEnabled);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView <see cref="IInkMARCDrawingView.DrawAction"/> property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapDrawAction(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetDrawAction(view.DrawAction);
    }

    /// <summary>
    /// Action that's triggered when the DrawingView Background property changes.
    /// </summary>
    /// <param name="handler">An instance of <see cref="AdvancedDrawingViewHandler"/>.</param>
    /// <param name="view">An instance of <see cref="IInkMARCDrawingView"/>.</param>
    public static void MapDrawingViewBackground(AdvancedDrawingViewHandler handler, IInkMARCDrawingView view)
    {
        handler.PlatformView.SetPaint(view.Background ?? new SolidPaint(CommunityToolkit.Maui.Core.DrawingViewDefaults.BackgroundColor));
    }

    /// <inheritdoc />
    protected override void ConnectHandler(InkMARCDrawingView platformView)
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
    protected override void DisconnectHandler(InkMARCDrawingView platformView)
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
	protected override InkMARCDrawingView CreatePlatformView() => new(Context);
#else
    protected override InkMARCDrawingView CreatePlatformView() => new();
#endif

    void OnPlatformViewDrawingLineCompleted(object? sender, InkMARCDrawingLineCompletedEventArgs e)
    {
        var drawingLine = adapter.ConvertInkMARCDrawingLine(e.Line);
        VirtualView.OnDrawingLineCompleted(drawingLine);
    }

    void OnPlatformViewDrawing(object? sender, InkMARCOnDrawingEventArgs e)
    {
        VirtualView.OnPointDrawn(e.Point);
    }

    void OnPlatformViewDrawingStarted(object? sender, InkMARCDrawingStartedEventArgs e)
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
